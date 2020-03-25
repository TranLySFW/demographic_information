from pathlib import Path
import os
import cv2
import dlib
import numpy as np
import argparse
import time 
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

# import multiprocessing
# import subprocess
# import datetime

# from contextlib import contextmanager
# from wide_resnet import WideResNet
# from model import get_model
# #from keras.utils.data_utils import get_file

# from omxplayer.player import OMXPlayer
# import logging


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

# def get_args():
#     parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
#                                                  "and estimates age and gender for the detected faces.",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--weight_file_gender", type=str, default=None,
#                         help="path to weight file (e.g. weights.28-3.73.hdf5)")
#     parser.add_argument("--depth", type=int, default=16,
#                         help="depth of network")
#     parser.add_argument("--width", type=int, default=8,
#                         help="width of network")
#     parser.add_argument("--margin", type=float, default=0.4,
#                         help="margin around detected face for age-gender estimation")
                        
#     parser.add_argument("--model_name", type=str, default="ResNet50",
#                         help="model name: 'ResNet50' or 'InceptionResNetV2'")
#     parser.add_argument("--weight_file_age", type=str, default=None,
#                         help="path to weight file (e.g. age_only_weights.029-4.027-5.250.hdf5)")
                        
#     args = parser.parse_args()
#     return args
    
    
def display_ads(status_age, status_gender, quitting):
    '''Display videos of advertising on the screen, 
       There will be a gap between switching times 
       Input: age, gender
       Output: none
    '''
    while True:
        if status_age.value == 1: 
            if status_gender.value > 0.5:
                list_files = subprocess.run(["omxplayer","-b","--display", "7", "-o", "local", "/home/pi/Downloads/1.mp4"], stdout=subprocess.DEVNULL)
            else:
                list_files = subprocess.run(["omxplayer","-b", "--display", "7", "-o", "local", "/home/pi/Downloads/2.mp4"], stdout=subprocess.DEVNULL)
        elif status_age.value == 2:
            if status_gender.value > 0.5:
                list_files = subprocess.run(["omxplayer","-b","--display", "7", "-o", "local", "/home/pi/Downloads/3.mp4"], stdout=subprocess.DEVNULL)
            else:
                list_files = subprocess.run(["omxplayer","-b", "--display", "7", "-o", "local", "/home/pi/Downloads/4.mp4"], stdout=subprocess.DEVNULL)
        elif status_age.value == 3:
            if status_gender.value > 0.5:
                list_files = subprocess.run(["omxplayer","-b","--display", "7", "-o", "local", "/home/pi/Downloads/5.mp4"], stdout=subprocess.DEVNULL)
            else:
                list_files = subprocess.run(["omxplayer","-b", "--display", "7", "-o", "local", "/home/pi/Downloads/6.mp4"], stdout=subprocess.DEVNULL)
        elif status_age.value == 4:
            if status_gender.value > 0.5:
                list_files = subprocess.run(["omxplayer","-b","--display", "7", "-o", "local", "/home/pi/Downloads/7.mp4"], stdout=subprocess.DEVNULL)
            else:
                list_files = subprocess.run(["omxplayer","-b", "--display", "7", "-o", "local", "/home/pi/Downloads/8.mp4"], stdout=subprocess.DEVNULL)
      

def player_wrapper(status_age, status_gender, quitting):
    '''Display videos of advertising on the screen
       Videos need to be pre-processed before using (use Video Editor on Windows 10)
       Input: age, gender
       Output: none
    '''
    # Concatenate videos to output.mp4
    # Duration of each videos in duration = [begin = 0, x1(second), x2(second), ...]
    # duration = [0, 22.89, 19.56, 15.85, 19.02, 21.28, 18.89, 17.47, 15.85]
    
    duration = [0, 15, 15, 15, 15, 15, 15, 15, 15] 
    VIDEO_PATH = Path("/home/pi/Downloads/print_ads.mp4")
    # Open instance of omxplayer
    # -o local for using sound from Raspberry Pi, for HDMI sound, please use -o hdmi
    player = OMXPlayer(VIDEO_PATH, args=["-b","--display", "7","--layer", "10", "-o", "local"])
    player.pause()
    
    def player_video(player, duration, video_ith):
        ''' Play specific video i in order
            input: player, list of durations, order
            output: none
        '''
        compensate = 1.2  #omxplayer-wrapper doesnt work accuracy at set_position, reduce the range of video
        alpha = 0.5       #keep player playing
        start_pos = sum(duration[:video_ith]) # start position of video ith
        duration_ith = duration[video_ith]
        player.set_position(start_pos + compensate) #use DBus to set position of video
        player.play()
        
        if video_ith == (len(duration) - 1): # the last video 
            time.sleep(duration_ith - compensate  - alpha) # remain player
        else:
            time.sleep(duration_ith - compensate)
            
        player.pause()
        return None
    
    while True:
       if status_age.value == 1: 
           if status_gender.value > 0.5: # display video 1
               player_video(player, duration, 1)
           else: # display video 2
               player_video(player, duration, 2)
       elif status_age.value == 2:
           if status_gender.value > 0.5:
               player_video(player, duration, 3)
           else:
               player_video(player, duration, 4)
       elif status_age.value == 3:
           if status_gender.value > 0.5:
               player_video(player, duration, 5)
           else:
               player_video(player, duration, 6)
       elif status_age.value == 4:
           if status_gender.value > 0.5:
               player_video(player, duration, 7)
           else:
               player_video(player, duration, 8)
      
    return None

    
def main(status_age, status_gender, quitting):
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
    cap = cv2.VideoCapture(-1) #use 0, 1, 2 if hardware raises error
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

                status_age.value = int(text_age)
                status_gender.value = gender_prob[0][0]

                content = "G: " + text_gender + ", R: " + text_age + ", ID: " + str(id_count)
                image = draw_label(image, top_left, bottom_right, content)  

        cv2.imshow("result", image)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()



# def main(status_age, status_gender, quitting):  
#     args = get_args() # get arguments
#     depth = args.depth
#     k = args.width
#     weight_file_gender = args.weight_file_gender #get gender model
#     weight_file_age = args.weight_file_age #get age model
#     model_name = args.model_name #Resnet50 or Xception
#     margin = args.margin
    
#     # for face detection
#     detector = dlib.get_frontal_face_detector()
    
#     # load model and weights
#     img_size_gender = 64
#     model_gender = WideResNet(img_size_gender, depth=depth, k=k)()
#     model_gender.load_weights(weight_file_gender)
    
#     model_age = get_model(model_name=model_name)
#     model_age.load_weights(weight_file_age)
#     img_size_age = model_age.input.shape.as_list()[1]
    
#     #adjust framerate from camera
#     frame_rate = 0.25
#     prev = 0
    
#     # capture video
#     cap = cv2.VideoCapture(-1) #use 0, 1, 2 if hardware raises error
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     while True:
#         time_elapsed = time.time() - prev #
#         ret, img = cap.read() # get video frame
#         img = cv2.rotate(img, cv2.ROTATE_180) # layout of camera on reletive positions
        
#         if time_elapsed > 1./frame_rate:
            
#             print("Get frame", str(datetime.datetime.now()))
#             prev =  time.time()
        
#             input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change to RGB
#             img_h, img_w, _ = np.shape(input_img) # get height and width of image

#             detected = detector(input_img, 1) #get faces 
            
#             faces_age = np.empty((len(detected), img_size_age, img_size_age, 3)) #number of faces captured
#             faces_gender = np.empty((len(detected), img_size_gender, img_size_gender, 3)) 

#             if len(detected) > 0: #at least one face is captured
#                 for i, d in enumerate(detected):
#                     x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
#                     xw1 = max(int(x1 - margin * w), 0)
#                     yw1 = max(int(y1 - margin * h), 0)
#                     xw2 = min(int(x2 + margin * w), img_w - 1)
#                     yw2 = min(int(y2 + margin * h), img_h - 1)
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
#                     faces_age[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size_age, img_size_age))
#                     faces_gender[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size_gender, img_size_gender))

                
#                 results_age = model_age.predict(faces_age)
#                 ages = np.arange(0, 101).reshape(101, 1)
#                 predicted_ages = results_age.dot(ages).flatten()
            
            
#                 results_gender = model_gender.predict(faces_gender)
#                 predicted_genders = results_gender[0]
#                 # ages = np.arange(0, 101).reshape(101, 1)
#                 # predicted_ages = results[1].dot(ages).flatten()
                
#                 #please noted that the code gets value from the first face in frame, not all of them
#                 #in the future: get the face has the largest bounding box (nearest)
#                 status_age.value = int(predicted_ages[0])
#                 status_gender.value = predicted_genders[0][0]
        
#                 print("Age", status_age.value)
#                 print("Gender", "Female" if status_gender.value > 0.5 else "Male")
                
#                 for i, d in enumerate(detected):
#                     label = "{}, {}".format(int(status_age.value),
#                                             "M" if status_gender.value < 0.5 else "F")
#                     draw_label(img, (d.left(), d.top()), label)

#             cv2.imshow("result", img)
#             key = cv2.waitKey(30)

#             if key == 27:  # ESC
#                 break
    
#     cap.release()
#     cv2.destroyAllWindows()
                


        
if __name__ == '__main__':
    status_gender = multiprocessing.Value("d", 1) #share value between processes
    status_age = multiprocessing.Value("d",25)
    quitting = multiprocessing.Value("d", 1)
    
    p1 = multiprocessing.Process(target=main, args=(status_age, status_gender, quitting,))
    p2 = multiprocessing.Process(target=player_wrapper, args=(status_age, status_gender, quitting,))

    p1.start()
    p2.start()
 
    while input(">Press Q to exit") != 'q':
        pass
    
    for p1 in multiprocessing.active_children():
        p1.terminate()
        
    for p2 in multiprocessing.active_children():
        p2.terminate()
        
    os.system("killall omxplayer.bin")
    
    p1.join()
    p2.join()
   
    
