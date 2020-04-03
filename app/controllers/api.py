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
import tensorflow as tf

# Compare between previous and current frame, it wont increase if there is no other person
identification = 0
previous_gender = 0
previous_age = tf.zeros(shape=(1, 8), dtype=tf.float32)
previous_vector = tf.zeros(shape=(1, 128), dtype=tf.float32)


@app.route('/api/capture', methods=['GET', 'POST'])
def capture_frame():
    """
    Capture one frame and predict
    :param :
    :return: json file
    """
    global identification, previous_vector, previous_age, previous_gender
    # ID verification
    diff_threshold = -0.8  # Range[-1, 0]
    diff_age_thres = -0.8  # Range[-1, 0]
    diff_gender_thres = 0.05 # should in range[0.01, 0.3]

    # adjust framerate from camera
    frame_rate = 25
    prev = 0
    alpha = 1.5

    ret, image = cap.read()  # get video frame

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    predict.net.setInput(blob)
    detections = predict.net.forward()
    detected, crop_face, top_left, bottom_right = streaming.nearest_standing(image, detections, alpha)

    content = {}
    if detected:
        age_prob = predict.predict_age_id(crop_face)
        # print("Age: " , np.argmax(age_prob))
        gender_prob = predict.predict_gender(crop_face)
        # print("Gender: ", gender_prob)
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
                identification += 1
        else:
            if detected_age_value > -1 - diff_age_thres:
                identification += 1

        previous_vector = vector_face
        previous_age = age_prob[0]
        previous_gender = gender_prob[0][0]

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
        diff_threshold = -0.8  # Range[-1, 0]
        diff_age_thres = -0.6  # Range[-1, 0]
        diff_gender_thres = 0.05  # should in range[0.01, 0.3]

        previous_gender = 0
        previous_age = tf.zeros(shape=(1, 8), dtype=tf.float32)
        previous_vector = tf.zeros(shape=(1, 128), dtype=tf.float32)
        id_count = 0

        # adjust framerate from camera
        frame_rate = 25
        prev = 0
        alpha = 1.5

        while True:
            time_elapsed = time.time() - prev  #
            ret, image = cap.read()  # get video frame
            if time_elapsed > 1. / frame_rate:
                prev = time.time()
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                predict.net.setInput(blob)
                detections = predict.net.forward()
                detected, crop_face, top_left, bottom_right = streaming.nearest_standing(image, detections, alpha)

                content = {
                        'gender': 'X',
                        'age': 'X',
                        'id': 'X',
                        'vector': 'X',
                }

                if detected:
                    age_prob = predict.predict_age_id(crop_face)
                    # print("Age: " , np.argmax(age_prob))
                    gender_prob = predict.predict_gender(crop_face)
                    # print("Gender: ", gender_prob)
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

