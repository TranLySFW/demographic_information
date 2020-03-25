![](https://i.imgur.com/HlxKXDX.png)

# Introduction 

Demographic information is becoming more and more influential in advertising industry nowsaday. Customers have certain buying patterns based on their ages and genders. For example, the young group pays attention in technology products ,otherwise the older group spend more on health care products and pharmaceuticals. This information helps the companies localize themselves, focusing at specific groups of population.

In the other way, customers will also be received their advantages. They won't be supplied inappropriate advertisements. 

The questions is, how can we use demographic analysing in advertisements but keeping privacy of our customers. And AI algorithms is developed to supply a solution.

# Table of Content

[Toc]

# 1. Goal

![](https://i.imgur.com/YPTZVuz.jpg)

My client is an business' owner at France. They would like to know about age and gender to display suitable advertisements. And the law requires that you don't be allowed to collect or storing information of customers. 

In that situation, an end device will run age and gender detection based on a deep learning model and displaying exist ads in hard drive. No internet required. 

In the case, system requires an online detector. The system control will call API to our server on Google App Engine.

# 2. System

![](https://i.imgur.com/4hRqXeM.png)

- Hardware: it can be Raspberry Pi, NUC, Jetson, ... 
- System Control: system of client which control other functions, also displaying function.
- Camera module: Pi Camera or any USB Cameras
- Google App Engine: deployed through Google Cloud platform
- Flask: API Server

An hardware(Pi4) with camera module will capture faces of people are standing facing it. The model will detect age and gender of those people. A restful API server is built on this hardware system will stream API responses to System Control(SC). Based on information from SC, it will display suitable ads.

In case SC want to call API online to our Google App Engine(GAE), then this server will return result back to SC.


# 3. Hardware

## 3.1 Processing hardware
* 1 Raspberry Pi 4 Model B 4GB RAM at [here](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
* 1 Camera module [here](https://www.raspberrypi.org/products/camera-module-v2/)
* 1 microSD card 16GB
* 1 charger
* Maximum power is 15W (comparing with a light buld ~ 40W)

## 3.2 System Control
Client will build their own system. It communicates through API calls with our hardware

# 4. Software

## 4.1. OS: Raspian Buster [here](https://www.raspberrypi.org/downloads/raspbian/)
You can refer from main page of Raspberry Pi to install OS effectively (with monitor or headless install)

## 4.2. OpenCV

### 4.2.1. Install neccesary libraries
* Before proceeding, we should first update any preexisting packages.
```bash=
sudo apt update
sudo apt upgrade
```
* Now we can start the process of installing all the packages we need for OpenCV to compile.
```bash=
sudo apt install cmake build-essential pkg-config git
```
* Next, we are going to install the packages that will add support for different image and video formats to OpenCV.
```bash=
sudo apt install libjpeg-dev libtiff-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
```
* Our next step is to install all the packages needed for OpenCV’s interface by using the command below.
```bash=
sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
```
* These next packages are crucial for OpenCV to run at a decent speed on the Raspberry Pi.
```bash=
sudo apt install libatlas-base-dev liblapacke-dev gfortran
```
* The second last lot of packages thaat we need to install relate to the Hierarchical Data Format (HDF5) that OpenCV uses to manage data.Install the HDF5 packages to your Pi by using the command below. If you have error with libhdf5-103 lib, using libhdf5-100 instead.
```bash=
sudo apt install libhdf5-dev libhdf5-103
```
* Finally, we can install the final few packages by using the command below.These last few packages will allow us to compile OpenCV with support for Python on our Raspberry Pi.
```bash=
sudo apt install python3-dev python3-pip python3-numpy
```
### 4.2.2. Preparing your Raspberry Pi for Compiling OpenCV

* The swap space is used by the operating system when the device has run out of physical RAM. While swap memory is a lot slower than RAM, it can still be helpful in certain situations.Begin modifying the swap file configuration by running the following command.
```bash=
sudo nano /etc/dphys-swapfile
```
* While we are within this file, we need to find and replace CONF_SWAPIZE=2048

* As we have made changes to the swapfile configuration, we need to restart its service by utilizing the command below.
```bash=
sudo systemctl restart dphys-swapfile
```
*  Next, let’s go ahead and clone the two OpenCV repositories we need to our Raspberry Pi.
```bash=
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

### 4.2.3. Compiling OpenCV on your Raspberry Pi
* Let’s start by creating a directory called “build” within the cloned “opencv” folder and then changing the working directory to it.
```bash=
mkdir ~/opencv/build
cd ~/opencv/build
```
* Now that we are within our newly created build folder, we can now use cmake to prepare OpenCV for compilation on our Raspberry Pi.Run the following command to generate the required makefile.
```bash=
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
    -D BUILD_EXAMPLES=OFF ..
```
* Once the make file has successfully finished generating, we can now finally move on to compiling OpenCV by running the command below (about 1 hour)
```bash=
make -j4
```
* When the compilation process finishes, we can then move on to installing OpenCV.
```bash=
sudo make install
```

* Now we also need to regenerate the operating systems library link cache.
```bash=
sudo ldconfig
```

### 4.2.4. Cleaning up after Compilation

* Changing CONF_SWAPIZE=100 as upper instruction

### 4.2.5. Testing OpenCV on your Raspberry Pi
```bash=
python3
import cv2
cv2.__version__
```

## 4.3. Tensorflow **2.x**

* Installing necessary libs
```bash=
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
python3 -m pip install keras_applications==1.0.8 --no-deps
python3 -m pip install keras_preprocessing==1.1.0 --no-deps
python3 -m pip install h5py==2.9.0
sudo apt-get install -y openmpi-bin libopenmpi-dev
sudo apt-get install -y libatlas-base-dev
python3 -m pip install -U six wheel mock
```
* Pick a tensorflow release from this link and download using wget
[Tensorflow](https://github.com/lhelontra/tensorflow-on-arm/releases)
```bash=
wget https://github.com/lhelontra/tensorfl...
```

* Install version 2
```bash=
python3 -m pip uninstall tensorflow
python3 -m pip install tensorflow-2.0.0-cp37-none-linux_armv7l.whl
```

* Testing
```bash=
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

## 4.4 Flask
```bash=
pip3 install flask
```
## 4.5 Environment
There are dependencies package in requirements.txt


## 4.4. omxplayer-wrapper [link](https://python-omxplayer-wrapper.readthedocs.io/en/latest/)
If client doenn't want to use their own System Control. The hardware will display ads through HDMI ports.

```bash=
pip3 install omxplayer-wrapper
```

# 5. Models

The process for creating models in saved in notebooks in notebook folder(ipynb and corresponding markdown):

- Age_Gender_Proprocess: cleaning datasets
- Training_Models: building architects and training
- Convert_Models: converting models to tensorflow (pb, pbtxt) or tensorflow lite (tflite) formats
- Run_Models: testing models with images and camera on Jupyter Notebook

All models are saved in script_models and they won't be published. Please contact me if you are interesting in this project [tranlysfw@gmail.com]

There are one model for facial detection, one for age, one for gender predictions.


# 6. Flask API server

## 6.1 Deploy on Hardware
- Change directory to app folder
```bash=
cd ./app
```

- Run main.py
```bash=
python main.py
```

## 6.2 Deploy on Google App Engine
- At first place, I've chosen the GAE because it's easy to deploy. But for long term, we should use Google VM, it has better price.
- Install [Google Cloud SDK](https://cloud.google.com/sdk)
- Authorization 
```bash=
gcloud auth login
```
- Creat new project
```bash=
gcloud app create --project=[.....] --region=[....]
```
- Get list of projects
```bash=
gcloud projects list
```
- Set project to work with
```bash=
gcloud config set project my-project
```
- Confirm project we are working with
```bash=
gcloud config get-value project
```
- Change directory to app and run
```bash=
gcloud app deploy
```
- After deploying, we can check web app
```bash=
gcloud app browse
```
- Review log file if errors happened
```bash=
gcloud app logs read
```

## 6.3 End points and Restful API

### 6.3.1 Image only

We can upload image and predict age and gender

**http://localhost:8080**

![](https://i.imgur.com/xpoRI9W.jpg)

![](https://i.imgur.com/uf2E1qs.jpg)

### 6.3.2 Camera streaming

It's streaming continously from camera to screen

**http://localhost:8080/streaming**

![](https://i.imgur.com/zBPmaf0.png)

### 6.3.3 Capture only one frame and predict

Call restful API, camera will capture one frame for processing

**http://localhost:8080/api/capture**

![](https://i.imgur.com/y0xwSmN.png)

### 6.3.4 Stream results only with API

Streaming Responses continuously

**http://localhost:8080/api/stream**

![](https://i.imgur.com/lCKnFlK.png)

### 6.3.5 Stream result and save to database

We are using SQLite to save data and it will be used to extract data

**http://localhost:8080/api/stream-and-write**
![](https://i.imgur.com/wfGD8pl.png)


# 7. Runing without System Control

## 7.1 Create videos

* Open Video Editor on Windows 10

![](https://i.imgur.com/8N3e7mb.png)

* Add more videos and putting it in Storyboard

![](https://i.imgur.com/t3Qz7ht.jpg)

* Copy duration of each videos in demo.py

```python=
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
```


## 7.2 Run on Raspberry Pi with HDMI output

* Change directory to downloaded folder
```bash=
cd <code_folder>
```
* Open demo.py, change VIDEO_PATH value to print_ads.mp4

```python=
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
```

* run command line
```bash=
python3 demo.py
```


# 8. Important note

This is a real project, models in folder are baseline ones. The true models won't be published to anyone except client who pays for this project.









