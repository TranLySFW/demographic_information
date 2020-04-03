from main import app  # Use when upload to Google App Engine

import time
import os
import cv2
import base64
import pathlib
import numpy as np
import multiprocessing
import tensorflow as tf

# from tensorflow.keras.models import load_model

age_model_path = pathlib.Path(os.path.join(app.root_path, "models/xception_age_id.tflite"))  # path to age model
age_model = tf.lite.Interpreter(model_path=str(age_model_path.absolute()))
age_model.allocate_tensors()
input_age_details = age_model.get_input_details()
output_age_details = age_model.get_output_details()

gender_model_path = pathlib.Path(os.path.join(app.root_path, "models/xception_gender.tflite"))  # path to gender model
gender_model = tf.lite.Interpreter(model_path=str(gender_model_path.absolute()))
gender_model.allocate_tensors()
input_gender_details = gender_model.get_input_details()
output_gender_details = gender_model.get_output_details()

age_queue = multiprocessing.Queue()
gender_queue = multiprocessing.Queue()

proto = os.path.join(app.root_path, "models", "deploy.prototxt")
caffe = os.path.join(app.root_path, "models", "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(proto, caffe)


def preprocess_image(image):
    """
    Import image, pre-process it before push it in model
    :param image:
    :return: processed image
    """
    # de-noise parameter, higher is stronger
    denoise = 5
    processed_img = cv2.fastNlMeansDenoisingColored(image, None, denoise, 10, 7, 21)
    ## change image to HSV color space to brighten it up
    # hsvImg = cv2.cvtColor(processed_img, cv2.COLOR_RGB2HSV)
    # value = 50
    # vValue = hsvImg[..., 2]
    # hsvImg[..., 2] = np.where((255 - vValue) < value, 255, vValue + value)
    # processed_img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB)
    resized = cv2.resize(processed_img, (299, 299),
                         interpolation=cv2.INTER_AREA)  # resize image to 299x299, input of Xception model
    resized = np.expand_dims(resized, axis=0)
    img = resized / 255.
    return img


def face_detector(image):
    """
    Use opencv dnn, import caffe model to detect human faces, crop it
    :param image:
    :return: list of cropped images
    """
    global net
    alpha = 1.5  # ratio border around human face, the larger it is, the more background it gets
    confidence_para = 0.5  # set confidence of model
    height, width, channel = image.shape
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    face_counts = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # extract the confidence (i.e., probability) associated with the prediction

        if confidence > confidence_para:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            left, top, right, bottom = box.astype("int")

            center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)  # get center location
            border = int((right - left) * alpha)  # border around face with ratio alpha

            x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
            x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)

            if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
                cropped_face = image[y_up: y_up + border, x_left: x_left + border]
                face_counts.append(cropped_face)

    return face_counts, detections  # return list of cropped faces in image


def nparray_to_base64(np_image):
    """
    :param np_image: decoded images and in numpy format height, width, channel
    :return: base64 format to display on html <img>
    """
    _, img_encode = cv2.imencode('.jpg', np_image)
    b64_string_output = base64.b64encode(img_encode)  # convert jpeg to base64
    chained_img = b'data:image/jpeg;base64,' + b64_string_output  # https://en.wikipedia.org/wiki/Data_URI_scheme#Syntax
    chained_img = chained_img.decode("utf-8")
    return chained_img


def predict_age(image):
    img = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(img, np.float32)
    age_model.set_tensor(input_age_details[0]['index'], input_tensor)
    age_model.invoke()
    probability_age = age_model.get_tensor(output_age_details[1]['index'])
    return probability_age


def predict_gender(image):
    img = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(img, np.float32)
    gender_model.set_tensor(input_gender_details[0]['index'], input_tensor)
    gender_model.invoke()
    probability_gender = gender_model.get_tensor(output_gender_details[0]['index'])
    return probability_gender


def predict_age_id(image):
    img = preprocess_image(image)
    input_tensor = tf.convert_to_tensor(img, np.float32)
    age_model.set_tensor(input_age_details[0]['index'], input_tensor)
    age_model.invoke()
    id_age = age_model.get_tensor(output_age_details[0]['index'])
    probability_age = age_model.get_tensor(output_age_details[1]['index'])
    return (probability_age, id_age)
