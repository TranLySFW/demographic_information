from main import app, cap  # Use when upload to Google App Engine
from controllers import predict
from controllers import streaming
from flask import Response, jsonify, stream_with_context

import os
import cv2
import numpy as np
import json
import time
import datetime
import face_recognition
import tensorflow as tf

# Compare between previous and current frame, it wont increase if there is no other person
identification = 0
previous_id = np.zeros(shape=(1, 128))


@app.route('/api/capture', methods=['GET', 'POST'])
def capture_frame():
    """
    Capture one frame and predict
    :param :
    :return: json file
    """
    global identification, previous_id
    # ID verification
    detected_id_threshold = -0.9

    frame_rate = 25  # adjust framerate from camera
    prev = 0
    alpha = 1.5

    ret, image = cap.read()  # get video frame

    _, detections = predict.face_detector(image)  # detect face in a picture
    detected, crop_face, top_left, bottom_right = streaming.nearest_standing(image, detections, alpha)

    content = {}
    if detected:
        vector_face = face_recognition.face_encodings(crop_face)
        if vector_face:
            age_prob = predict.predict_age_id(crop_face)
            gender_prob = predict.predict_gender(crop_face)
            text_gender = "M" if gender_prob[0][0] > 0.5 else "F"
            text_age = str(np.argmax(age_prob[0]))

            detected_id = tf.keras.losses.cosine_similarity(previous_id, vector_face[0]).numpy()
            if detected_id > detected_id_threshold:
                identification += 1
            previous_id = vector_face[0]
            # content = "G: " + text_gender + ", R: " + text_age + ", ID: " + str(identification)
            content = {
                'gender': text_gender,
                'age': text_age,
                'id': identification,
                'time': datetime.datetime.now(),
                'vector': vector_face[0].tolist(),
            }
    else:
        pass  # content is a blank dictionary
    print(jsonify(content))
    return jsonify(content)


@app.route('/api/stream', methods=['GET', 'POST'])
def api_stream():
    def generate_api_stream():
        """
        Streaming results as json
        :param :
        :return: flow of video frame as json
        """
        # ID verification
        previous_id = np.zeros(shape=(1, 128))
        detected_id_threshold = -0.9
        id_count = 0

        frame_rate = 25  # adjust framerate from camera
        prev = 0
        alpha = 1.5

        while True:
            time_elapsed = time.time() - prev  #
            ret, image = cap.read()  # get video frame
            if time_elapsed > 1. / frame_rate:
                prev = time.time()

                _, detections = predict.face_detector(image)  # detect face in a picture
                detected, crop_face, top_left, bottom_right = streaming.nearest_standing(image, detections, alpha)

                content = {}
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

                        # content = "G: " + text_gender + ", R: " + text_age + ", ID: " + str(id_count)
                        content = {
                            'gender': text_gender,
                            'age': text_age,
                            'id': identification,
                            'time': datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
                            'vector': vector_face[0].tolist(),
                        }

                        if not ret:
                            print("Error: failed to capture image")
                            break

                        yield json.dumps(content)

    # return Response(stream_with_context(generate_api_stream()))
    return Response(stream_with_context(generate_api_stream()), mimetype="text/plain")
