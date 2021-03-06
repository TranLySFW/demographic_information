from main import app, cap  # Use when upload to Google App Engine
from controllers import predict
from flask import Response

import os
import cv2
import numpy as np
import time
import face_recognition
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
                border_nearest, center_x_nearest, center_y_nearest = border, center_x, center_y

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
    previous_id = np.zeros(shape=(1, 128))
    detected_id_threshold = -0.9
    id_count = 0

    frame_rate = 25  # adjust frame rate from camera
    prev = 0
    alpha = 1.5  # border of face

    while True:
        time_elapsed = time.time() - prev  #
        ret, image = cap.read()  # get video frame
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            _, detections = predict.face_detector(image)  # detect face in a picture
            detected, crop_face, top_left, bottom_right = nearest_standing(image, detections, alpha)

            content = ""
            if detected:
                vector_face = face_recognition.face_encodings(crop_face)
                if vector_face:
                    age_prob = predict.predict_age_id(crop_face)
                    gender_prob = predict.predict_gender(crop_face)
                    text_gender = "M" if gender_prob[0][0] > 0.5 else "F"
                    text_age = str(np.argmax(age_prob[0]))

                    detected_id = tf.keras.losses.cosine_similarity(previous_id, vector_face[0]).numpy()

                    if detected_id > detected_id_threshold:
                        id_count += 1
                    previous_id = vector_face[0]

                    content = "G: " + text_gender + ", R: " + text_age + ", ID: " + str(id_count)
                    image = draw_label(image, top_left, bottom_right, content)

            if not ret:
                print("Error: failed to capture image")
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')
