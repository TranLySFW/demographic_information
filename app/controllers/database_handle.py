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
import sqlite3
import glob
import pathlib
from sqlite3 import Error
import tensorflow as tf


@app.route('/api/stream-and-write', methods=['GET', 'POST'])
def api_stream_and_write():
    def setup_database():
        conn, path_to_db = create_connection() # setup new database
        create_table(conn) # create predictor table
        return conn, path_to_db

    def generate_api_stream():
        """
        Streaming results as json
        :param :
        :return: flow of video frame as json and write record to database
        """
        conn, path_to_db = setup_database()  # create new database sqlite
        record_count = 0  # counter of record is limited in one database
        record_max = 100000  # maximum of records in a table

        # ID verification
        diff_threshold = -0.1  # Range[-1, 0]
        previous_vector = tf.zeros(shape=(1, 128), dtype=tf.float32)  # zeros vector
        id_count = 0  # counter times of detecting other faces

        frame_rate = 25  # adjust framerate from camera
        prev = 0  # time counter
        alpha = 1.5  # border of faces

        while True:
            time_elapsed = time.time() - prev
            ret, image = cap.read()  # get video frame
            if time_elapsed > 1. / frame_rate:
                prev = time.time()

                record_count += 1  # record is written
                if record_count == record_max:  # reach limitation
                    conn, path_to_db = setup_database()  # setup new database
                    id_count = 0  # reset counter

                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                predict.net.setInput(blob)
                detections = predict.net.forward()
                detected, crop_face, top_left, bottom_right = streaming.nearest_standing(image, detections, alpha)

                content = {}
                if detected:
                    age_prob = predict.predict_age_id(crop_face)
                    gender_prob = predict.predict_gender(crop_face)

                    text_gender = "M" if gender_prob[0][0] > 0.5 else "F"
                    text_age = str(np.argmax(age_prob[0]))
                    vector_face = age_prob[1]

                    detected_id_value = tf.keras.losses.cosine_similarity(previous_vector, vector_face).numpy()[0]
                    if detected_id_value > diff_threshold:
                        id_count += 1
                    previous_vector = vector_face

                    # save content into dictionary
                    content = {
                            'gender': text_gender,
                            'age': text_age,
                            'id': id_count,
                            'vector': json.dumps(vector_face[0].tolist()),
                    }
                    # write record to database
                    create_record(conn, (content['gender'], content['age'], content['id'], content['vector']))

                    if not ret:
                        print("Error: failed to capture image")
                        break

                    yield json.dumps(content)  # streaming dictionary to front end

    return Response(stream_with_context(generate_api_stream()))


@app.route('/api/test-database', methods=['GET', 'POST'])
def test_database():
    conn, path_to_db = create_connection()
    create_table(conn)
    content = {
            'gender': "M",
            'age': "2",
            'id': 10,
            'vector': json.dumps([0, 1, 3, 4]),
    }
    create_record(conn, (content['gender'], content['age'], content['id'], content['vector']))
    create_record(conn, (content['gender'], content['age'], content['id'], content['vector']))
    select_all(conn)
    return "Done"


@app.route('/api/list-database', methods=['GET', 'POST'])
def list_db_file():
    db_folder_path = pathlib.Path(os.path.join(app.root_path, "database"))
    content = []
    for file in db_folder_path.glob("*.db"):
        content.append(file.name)
    return json.dumps(content)


def select_all(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictor")
    rows = cur.fetchall()
    for row in rows:
        print(row)


def create_record(conn, record):
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """

    sql = """INSERT INTO predictor(gender,age,identification,vector)
              VALUES(?,?,?,?)"""

    cur = conn.cursor()
    cur.execute(sql, record)
    conn.commit()
    return cur.lastrowid


def create_table(conn):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    sql_create_table = """CREATE TABLE IF NOT EXISTS predictor (
                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                            gender CHAR(5) ,
                                            age CHAR(5) ,
                                            identification INTEGER,
                                            vector TEXT NOT NULL, 
                                            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""

    try:
        c = conn.cursor()
        c.execute(sql_create_table)
    except Error as e:
        print(e)


def create_connection():
    """ create a database connection to a SQLite database """
    # create name of database based on time
    curr_time = datetime.datetime.now()
    year, month, day, hour, minute = curr_time.year, curr_time.month, curr_time.day, curr_time.hour, curr_time.minute
    database_name = str(year) + "_" + str(month) + "_" + str(day) + "_" + str(hour) + "_" + str(minute) + ".db"
    path_to_database = os.path.join(app.root_path, "database", database_name)

    conn = None
    try:
        conn = sqlite3.connect(path_to_database)
        print(sqlite3.version)
    except Error as e:
        print(e)

    return conn, path_to_database
