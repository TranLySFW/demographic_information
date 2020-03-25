from flask import Flask, render_template, redirect, url_for
import os
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = "age_gender_random_key_2141516"  # Flask secret key
# app.config['UPLOAD_FOLDER'] = 'upload'

#import camera
cap = cv2.VideoCapture(0)  # use 0, 1, 2 if hardware raises error

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Only accept two types of image extension


@app.route('/')
def main_index():
    return render_template('home.html')


from controllers.predict import *
from controllers.preprocess import *
from controllers.upload import *
from controllers.streaming import *
from controllers.api import *
from controllers.database_handle import *


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app.run(host='0.0.0.0', port=8080, debug=False)
