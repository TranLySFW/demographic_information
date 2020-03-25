
import os
import cv2
import argparse
import time 
import multiprocessing
import subprocess
import datetime
import pathlib
import numpy as np


import tensorflow as tf
from tensorflow.keras.models import load_model



def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gender_model_path", type=str, default="script_models/xception_gender.h5",
                        help="path to weight file")
    parser.add_argument("--age_model_path", type=str, default="script_models/xception_age_id.h5",
                        help="path to weight file")
    parser.add_argument("--image", type=bool, default=False)                    
    args = parser.parse_args()
    return args
    
   
def draw_label(image, top_left, botom_right, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
	# size = cv2.getTextSize(label, font, font_scale, thickness)[0]
	cv2.rectangle(image, top_left, botom_right, (255, 0, 0), 1)
	cv2.putText(image, label, (top_left[0], top_left[1] - 15), font, font_scale, (255, 0, 0), 1, lineType=cv2.LINE_AA)
	return image


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
    resized = cv2.resize(processed_img, (299, 299), interpolation=cv2.INTER_AREA) # resize image to 299x299, input of Xception model
    resized = np.expand_dims(resized, axis=0)
    img = resized / 255.
    return img


def nearest_standing(image, detections, alpha):
    """ Comment
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
                border_nearest = border
                center_x_nearest = center_x
                center_y_nearest = center_y
                
                
    x_right, y_up = int(center_x_nearest + border_nearest / 2), int(center_y_nearest - border_nearest / 2)
    x_left, y_down = int(center_x_nearest - border_nearest / 2), int(center_y_nearest + border_nearest / 2)
    
    detected_nearest_face = False
    nearest_face = 0
    if x_left > 0 and x_left + border_nearest < width and y_up > 0 and y_up + border_nearest < height:
        nearest_face = image[y_up: y_up + border_nearest, x_left: x_left + border_nearest]
        detected_nearest_face = True
    
    return detected_nearest_face, nearest_face, (x_left, y_up), (x_right, y_down)


       
        
if __name__ == '__main__':

	args = get_args()

	gender_model_path = os.path.join(os.getcwd(), args.gender_model_path)
	age_model_path = os.path.join(os.getcwd(), args.age_model_path)

	xception_gender_model = load_model(gender_model_path)
	xception_age_model = load_model(age_model_path)

	proto = os.path.join(os.getcwd(), "script_models\\deploy.prototxt")
	caffe = os.path.join(os.getcwd(), "script_models\\res10_300x300_ssd_iter_140000.caffemodel")

	net = cv2.dnn.readNetFromCaffe(proto, caffe)

	# ID verification
	diff_threshold = -0.3 # Range[-1, 0] 
	previous_vector = tf.zeros(shape=(1, 128), dtype=tf.float32)
	id_count = 0

	#adjust framerate from camera
	frame_rate = 25
	prev = 0
	alpha = 1.5
	# capture video
	cap = cv2.VideoCapture(0) #use 0, 1, 2 if hardware raises error
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	while True:
		time_elapsed = time.time() - prev #
		ret, image = cap.read() # get video frame
		
		if time_elapsed > 1./frame_rate:
			# print("Get frame", str(datetime.datetime.now()))
			prev =  time.time()
			#input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change to RGB
			blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
			net.setInput(blob)
			detections = net.forward()
			detected, crop_face, top_left, bottom_right = nearest_standing(image, detections, alpha)

			content = ""
			if detected:
				img_input = preprocess_image(crop_face)
				age_prob = xception_age_model.predict(img_input)
				# print("Age: " , np.argmax(age_prob))
				gender_prob = xception_gender_model.predict(img_input)
				# print("Gender: ", gender_prob)
				text_gender = "M" if gender_prob[0][0] > 0.5 else "F"
				text_age = str(np.argmax(age_prob[0]))

				vector_face = age_prob[1]
				detected_id_value = tf.keras.losses.cosine_similarity(previous_vector, vector_face).numpy()[0]
				text_id = "Unknown"

				if detected_id_value < diff_threshold:
					text_id = "Same"
				else:
					text_id = "Diff"
					id_count += 1

				previous_vector = vector_face

				content = "G: " + text_gender + ", R: " + text_age + ", ID: " + str(id_count)
				image = draw_label(image, top_left, bottom_right, content)  

		cv2.imshow("result", image)
		key = cv2.waitKey(30)
		if key == 27:  # ESC
			break

	cap.release()
	cv2.destroyAllWindows()

