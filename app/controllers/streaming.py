from main import app, cap  # Use when upload to Google App Engine
from controllers import predict
from flask import Response

import os
import cv2
import numpy as np
import time
import tensorflow as tf


@app.route('/streaming')
def streaming_result():
    return Response(main_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def draw_label(image, top_left, botom_right, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
    # size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    cv2.rectangle(image, top_left, botom_right, (255, 0, 0), 1)
    cv2.putText(image, label, (top_left[0], top_left[1] - 15), font, font_scale, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    return image


def nearest_standing(image, detections, alpha):
    """ Find the maximum anchor box in picture to identify the nearest person
    """
    border_nearest, center_x_nearest, center_y_nearest = 0, 0, 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        height, width, channel = image.shape
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            left, top, right, bottom = box.astype("int")
            center_x = int((right + left) / 2)
            center_y = int((top + bottom) / 2)
            border = int((right - left) * alpha)

            if border > border_nearest:
                border_nearest, center_x_nearest, center_y_nearest  = border, center_x, center_y

    x_right, y_up = int(center_x_nearest + border_nearest / 2), int(center_y_nearest - border_nearest / 2)
    x_left, y_down = int(center_x_nearest - border_nearest / 2), int(center_y_nearest + border_nearest / 2)

    detected_nearest_face = False
    nearest_face = 0
    if x_left > 0 and x_left + border_nearest < width and y_up > 0 and y_up + border_nearest < height:
        nearest_face = image[y_up: y_up + border_nearest, x_left: x_left + border_nearest]
        detected_nearest_face = True

    return detected_nearest_face, nearest_face, (x_left, y_up), (x_right, y_down)


def main_stream():
    """
    Streaming results to image tag
    :param :
    :return: flow of video frame
    """
    # ID verification
    diff_threshold = -0.8  # Range[-1, 0]
    diff_age_thres = -0.9  # Range[-1, 0]
    diff_gender_thres = 0.05 # should in range[0.01, 0.3]

    previous_gender = 0
    previous_age = tf.zeros(shape=(1, 8), dtype=tf.float32)
    previous_vector = tf.zeros(shape=(1, 128), dtype=tf.float32)
    id_count = 0

    frame_rate = 25 # adjust framerate from camera
    prev = 0
    alpha = 1.5 # border of face

    while True:
        time_elapsed = time.time() - prev  #
        ret, image = cap.read()  # get video frame
        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            # detect face in a picture
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            predict.net.setInput(blob)
            detections = predict.net.forward()
            detected, crop_face, top_left, bottom_right = nearest_standing(image, detections, alpha)

            content = ""
            if detected:
                age_prob = predict.predict_age_id(crop_face)
                gender_prob = predict.predict_gender(crop_face)
                text_gender = "M" if gender_prob[0][0] > 0.5 else "F"
                text_age = str(np.argmax(age_prob[0]))

                vector_face = age_prob[1]
                detected_gender_value = abs(gender_prob[0][0] - previous_gender)
                detected_age_value = tf.keras.losses.cosine_similarity(previous_age, age_prob[0]).numpy()[0]
                detected_id_value = tf.keras.losses.cosine_similarity(previous_vector, vector_face).numpy()[0]
                text_id = "Unknown"
                if detected_gender_value > diff_gender_thres:
                    # print("Pass gender")
                    if detected_age_value > diff_age_thres:
                        # print("Pass age")
#                       if detected_id_value > diff_threshold:
                        id_count += 1
                else:
                    if detected_age_value > -1 - diff_age_thres:
                        id_count += 1

                previous_vector = vector_face
                previous_age = age_prob[0]
                previous_gender = gender_prob[0][0]

                content = "G: " + text_gender + ", R: " + text_age + ", ID: " + str(id_count)
                image = draw_label(image, top_left, bottom_right, content)

            if not ret:
                print("Error: failed to capture image")
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')
