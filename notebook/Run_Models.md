# Age model 

## Load model 


```python
age_model_path = "D:\\01_PYTHON\\05_CoderS\\23_Age_Gender_Prediction_VM\\script_models\\xception_age.h5"
```


```python
xception_age_model = load_model(age_model_path)
```

## Image 

Preprocess image


```python
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
```

Image folder path


```python
image_list_path = pathlib.Path("D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience\\datasets\\IMDB\\wiki_crop\\20")
```

Get file names of all jpeg files in folder


```python
image_list = list(image_list_path.glob("*.jpg"))
```

Read one image in list


```python
img = cv2.imread(str(image_list[63].absolute()))
```

**Convert image to RGB because imread function of CV2 always return with BGR format**


```python
img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
```

Test image 


```python
plt.imshow(img)
```

**predict**


```python
img_input = preprocess_image(img)
output = xception_age_model.predict(img_input)
```

Result


```python
output_class = np.argmax(output[0])
output_class
```

## Camera 


```python
#adjust framerate from camera
frame_rate = 10
prev = 0
alpha = 1.5
# capture video
cap = cv2.VideoCapture(0) #use 0, 1, 2 if hardware raises error
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    time_elapsed = time.time() - prev #
    ret, img = cap.read() # get video frame
#     img = cv2.rotate(img, cv2.ROTATE_180) # layout of camera on reletive positions

    if time_elapsed > 1./frame_rate:

        print("Get frame", str(datetime.datetime.now()))
        prev =  time.time()

#         input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change to RGB
        height, width, channel = input_img.shape  
        face_locations = face_recognition.face_locations(input_img)
        
        for face in face_locations:
            # print("face detected")
            top, right, bottom, left = face  
            center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)
            border = int((right - left) * alpha)

            x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
            x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)

            if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
                crop_face = img[y_up: y_up + border, x_left: x_left + border]
                img_input = pre_processing_image(crop_face)
                output = xception_age_model.predict(img_input)
                print(output)
                
#             for i, d in enumerate(detected):
#                 label = "{}, {}".format(int(status_age.value),
#                                         "M" if status_gender.value < 0.5 else "F")
#                 draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
```

<br>

# Gender model 

## Load model 


```python
gender_model_path = "D:\\01_PYTHON\\05_CoderS\\23_Age_Gender_Prediction_VM\\script_models\\xception_gender.h5"
```


```python
xception_gender_model = load_model(gender_model_path)
```

## Image 

Preprocess image


```python
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
```

Image folder path


```python
image_list_path = pathlib.Path("D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience\\datasets\\IMDB\\wiki_crop\\20")
```

Get file names of all jpeg files in folder


```python
image_list = list(image_list_path.glob("*.jpg"))
```

Read one image in list


```python
img = cv2.imread(str(image_list[63].absolute()))
```

**Convert image to RGB because imread function of CV2 always return with BGR format**


```python
img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
```

Test image 


```python
plt.imshow(img)
```

**predict**


```python
img_input = preprocess_image(img)
output = xception_gender_model.predict(img_input)
```

Result


```python
output
```

## Camera 


```python
#adjust framerate from camera
frame_rate = 10
prev = 0
alpha = 1.5
# capture video
cap = cv2.VideoCapture(0) #use 0, 1, 2 if hardware raises error
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    time_elapsed = time.time() - prev #
    ret, img = cap.read() # get video frame
#     img = cv2.rotate(img, cv2.ROTATE_180) # layout of camera on reletive positions

    if time_elapsed > 1./frame_rate:

        print("Get frame", str(datetime.datetime.now()))
        prev =  time.time()

#         input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change to RGB
        height, width, channel = input_img.shape  
        face_locations = face_recognition.face_locations(input_img)
        
        for face in face_locations:
            # print("face detected")
            top, right, bottom, left = face  
            center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)
            border = int((right - left) * alpha)

            x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
            x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)

            if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
                crop_face = img[y_up: y_up + border, x_left: x_left + border]
                img_input = pre_processing_image(crop_face)
                output = xception_gender_model.predict(img_input)
                print(output)
                
#             for i, d in enumerate(detected):
#                 label = "{}, {}".format(int(status_age.value),
#                                         "M" if status_gender.value < 0.5 else "F")
#                 draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
```


```python

```

# Run tensorflow model with OpenCV DNN

## Load model 


```python
model_path = ".../..pb" 
```


```python
cvnet = cv2.dnn.readNetFromTensorflow(model_path)
```

## Preprocess


```python
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
```

Get image


```python
img = cv2.imread("D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience\\datasets\\UTKface\\utkface\\gender\\male\\45_0_3_20170119201135740.jpg")
```

Test image


```python
plt.imshow(img)
```

Preprocess image


```python
img = preprocess_image(img)
```

## Predict 


```python
blob = cv2.dnn.blobFromImage(img,1,(299,299))
```


```python
cvnet.setInput(blob)
```


```python
cvout = cvnet.forward()
```


```python
cvout
```

<br>

# Face detection with OpenCV DNN instead of dlib

## Load model 


```python
proto = "D:\\01_PYTHON\\05_CoderS\\23_Age_Gender_Prediction_VM\\app\\models\\deploy.prototxt"
caffe = "D:\\01_PYTHON\\05_CoderS\\23_Age_Gender_Prediction_VM\\app\\models\\res10_300x300_ssd_iter_140000.caffemodel"
```


```python
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto, caffe)
```

## Preprocess image 


```python
image_path = "C:\\Users\\Admin\\Pictures\\Camera Roll\\WIN_20200209_21_01_04_Pro.jpg"
```


```python
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
```


```python
plt.imshow(image)
```

## Predict  


```python
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
```


```python
# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()
```


```python
# loop over the detections
alpha = 1.5
frame = image
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    height, width, channel = image.shape
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        left, top, right, bottom = box.astype("int")  
        center_x = int((right + left) / 2)
        center_y = int((top + bottom) / 2)
        border = int((right - left) * alpha)
        x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
        x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)
        if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
            crop_face = image[y_up: y_up + border, x_left: x_left + border]
```


```python
plt.imshow(crop_face)
```

<br>

# Identification 

## Preprocess image


```python
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
```

## Draw label function


```python
def draw_label(image, top_left, botom_right, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
    # size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    cv2.rectangle(image, top_left, botom_right, (255, 0, 0), 1)
    cv2.putText(image, label, (top_left[0], top_left[1] - 15), font, font_scale, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    return image
```

## Find the nearest person standing


```python
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
```

## **Run camera**


```python
args = get_args()
#Paths to models
gender_model_path = os.path.join(os.getcwd(), args.gender_model_path)
age_model_path = os.path.join(os.getcwd(), args.age_model_path)
#Load model
xception_gender_model = load_model(gender_model_path)
xception_age_model = load_model(age_model_path)
#Paths to face recognition
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
```
