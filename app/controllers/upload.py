from main import app  # Use when upload to Google App Engine
from controllers import predict

from flask import render_template, send_from_directory, request, flash, redirect, url_for, make_response
from werkzeug.utils import secure_filename
import numpy as np
import multiprocessing
import cv2
import math

# UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    """
    :param filename: get filename in ALLOWED_EXTENSIONS
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded(filename):
    """
    :param filename: upload filename in UPLOAD_FOLDER
    :return:
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    """
    :return: render home.html with results
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template("home.html")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_base64, results = main(file)
            return render_template("home.html", results=results, image=image_base64, filename=filename, label=1)
    else:
        print("GET method")


def main(image_file):
    """
    Get a uploaded file, crop faces, assign to result
    :param image_file: input file
    :return: list of human faces and their hyperparameters and original image in base64 format to display
    """
    img_input = np.fromstring(image_file.read(), np.uint8)  # read file to input stream
    img = cv2.imdecode(img_input, cv2.IMREAD_COLOR)  # decode input stream to numpy array

    original_img_base64 = predict.nparray_to_base64(img)  # return original image with bas64 format

    results = []
    detected_faces = predict.face_detector(img)  # detect faces in image

    if detected_faces:
        for i, face in enumerate(detected_faces):
            cropped_face = predict.nparray_to_base64(face)  # export to tag <img> in html

            prob_age = predict.predict_age(face)  # call model to predict age
            age_data = [math.floor(100 * x) for x in prob_age[0]]  # change probabilities to percentages
            grouped_age = np.argmax(prob_age[0])  # get the highest probability in prediction

            if grouped_age == 0:  # split age in range
                displayed_age = "0 ~ 10 years old"
            else:
                displayed_age = str(grouped_age) + "0" + " ~ " + str(grouped_age) + "9" + " years old"

            prob_gender = predict.predict_gender(face)  # call model to predict gender
            if prob_gender > 0.5:
                displayed_gender = "Male"
            else:
                displayed_gender = "Female"
            male_prob = math.floor(100 * prob_gender[0])  # change to percentages
            female_prob = 100 - male_prob

            results.append({'index_gender': 'canvas_gender_' + str(i),
                            'index_age': 'canvas_age_' + str(i),
                            'cropped_image': cropped_face,
                            'target_gender': displayed_gender,
                            'target_age': displayed_age,
                            'prob_male': male_prob,
                            'prob_female': female_prob,
                            'prob_age': age_data,
                            })
    else:
        print("Error: cannot detect any human faces in the picture")

    return original_img_base64, results
