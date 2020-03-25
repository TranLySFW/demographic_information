![](https://i.imgur.com/HlxKXDX.png)

# Introduction 

Demographic information is becoming more and more influential in advertising industry nowsaday. Customers have certain buying patterns based on their ages and genders. For example, the young group pays attention in technology products ,otherwise the older group spend more on health care products and pharmaceuticals. This information helps the companies localize themselves, focusing at specific groups of population.

In the other way, customers will also be received their advantages. They won't be supplied inappropriate advertisements. 

The questions is, how can we use demographic analysing in advertisements but keeping privacy of our customers. And AI algorithms is developed to supply a solution.

<br>

# Dataset 

I'm using three main dataset, all of these datasets has been downloaded from Kaggle and Internet.  

- [IMDB & Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- [UTKFace](https://susanqq.github.io/UTKFace/)
- [Asian faces](https://github.com/JingchunCheng/All-Age-Faces-Dataset)

There are a lot of unuseful images, and we need to clean it first.

<br>

# Environment

Models are trainned by using PC, Google Colab or Cloud Virtual Machine.

## Import libraries

###  Tensorflow 2


```python
from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
print(tf.__version__)
```

### other libs

Install **dlib, face_recogintion** to detect human faces for the first time. From the second one, we should comment these pip commands.


```python
# !pip3 install dlib
```


```python
# !pip3 install face_recognition
```


```python
import cv2  #openCV on python
import dlib #library for facial detection
import face_recognition #wrapper of dlib
import os
import shutil
import pathlib
import matplotlib.pyplot as plt #cv2.imshow error on GG colab and we use alternative of plt
import numpy as np
import pandas as pd 
import time 
import glob
```

## Google colab

Colaboratory, or "Colab" for short, allows you to write and execute Python in your browser, with 
- Zero configuration required
- Free access to GPUs
- Easy sharing

Mounting Google Drive to Colab for storing files


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

**Check GPU** existed or not


```python
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

**Checking CUDA version** with GG colab


```python
# This cell can be commented once you checked the current CUDA version
# CUDA: Let's check that Nvidia CUDA is already pre-installed and which version is it. In some time from now maybe you 
!/usr/local/cuda/bin/nvcc --version
```

**Install cuDNN according to the current CUDA version**


```python
# We're unzipping the cuDNN files from your Drive folder directly to the VM CUDA folders
!tar -xzvf /content/gdrive/My\ Drive/Final_project/darknet/cuDNN/cudnn-10.0-linux-x64-v7.5.0.56.tgz -C /usr/local/
!chmod a+r /usr/local/cuda/include/cudnn.h

# Now we check the version we already installed. Can comment this line on future runs
!cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

<br>

# UTKFace 

## Unzip dataset

Redirect to the exact folder to unzip dataset


```python
cd /home/tranlysfw/age_gender_prediction/dataset/utkface
```


```python
unzip /home/tranlysfw/age_gender_prediction/dataset/utkface/utkface.zip
```

## Preprocess

We get *inthewild* folders with **three parts** to detect human faces. Repeting this process three times with different parts.


```python
utkface_folder_wild = pathlib.Path(os.path.join(project_path, "inthewild/part1")) #part1, part2, part3
```

Because each file of utkface dataset include age and gender of that identity. So we just need to get name of files and paths


```python
project_path = pathlib.Path('/home/tranlysfw/age_gender_prediction/dataset/utkface')
utk_preprocess = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/utkface")
utk_male = pathlib.Path(os.path.join(utk_preprocess), "gender\\male\\")
utk_female = pathlib.Path(os.path.join(utk_preprocess), "gender\\female\\")
```

<br>

## Split names into lists

Get all paths of files on utkface dataset and its name. Extracting age, gender and race of human in the picture based on file names.


```python
img_paths = []
utkface_age = []
utkface_gender = []
utkface_race = []

for elem in utkface_folder.glob("*.jpg"):
    img_paths.append(elem)
    filename_splited = elem.name.split("_")
    utkface_age.append(filename_splited[0])
    utkface_gender.append(filename_splited[1])
    utkface_race.append(filename_splited[2])
```

Assign it to gender


```python
img_paths = []
utkface_age = []
utkface_gender = []
utkface_race = []
i = 0
alpha = 1.5

for elem in utkface_folder_wild.glob("*.jpg"):
    img = cv2.imread(str(elem.absolute()))
    height, width, channel = img.shape
    
    face_locations = face_recognition.face_locations(img)
    
    for face in face_locations:
        top, right, bottom, left = face  
        center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)
        border = int((right - left) * alpha)
        
        x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
        x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)
        
        if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
            crop_face = img[y_up: y_up + border, x_left: x_left + border]
            resized = cv2.resize(crop_face, (224, 224), interpolation=cv2.INTER_AREA)
            
            filename_splited = elem.name.split("_")
            utkface_age = filename_splited[0]
            utkface_gender = filename_splited[1]
#             print(utkface_gender)
            
            if int(utkface_gender) == 0:
                path = pathlib.Path(os.path.join(utk_male), elem.name)
#                 print(path)
                cv2.imwrite(str(path), resized)
            elif int(utkface_gender) == 1:
                path = pathlib.Path(os.path.join(utk_female), elem.name)
#                 print(path)
                cv2.imwrite(str(path), resized)
    
    i += 1
    if i % 1000 == 0: 
        print(i)

```

Copy relevant files to age folder


```python
utk_age_path = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/utkface/age")
```


```python
i = 0
for elem in utk_female.glob("*.jpg"):
    filename_splited = elem.name.split("_")
    utkface_age = int(filename_splited[0])
    utkface_gender = int(filename_splited[1])
    
    path = ""
    if utkface_age <= 10:
        path= pathlib.Path(os.path.join(utk_age_path), "0")
    elif utkface_age > 10 and utkface_age <= 20:
        path= pathlib.Path(os.path.join(utk_age_path), "1")
    elif utkface_age > 20 and utkface_age <= 30:
        path= pathlib.Path(os.path.join(utk_age_path), "2")
    elif utkface_age > 30 and utkface_age <= 40:
        path= pathlib.Path(os.path.join(utk_age_path), "3")
    elif utkface_age > 40 and utkface_age <= 50:
        path= pathlib.Path(os.path.join(utk_age_path), "4")
    elif utkface_age > 50 and utkface_age <= 60:
        path= pathlib.Path(os.path.join(utk_age_path), "5")
    elif utkface_age > 60 and utkface_age <= 70:
        path= pathlib.Path(os.path.join(utk_age_path), "6")
    elif utkface_age > 70:
        path= pathlib.Path(os.path.join(utk_age_path), "7")
        
    shutil.copy(str(elem.absolute()), path)   
    
    if i == 10:
        break
    
    
```

<br>

# Wiki

The dataset is great for research purposes. It contains more than 500 thousand+ images of faces. But the dataset is not ready for any Machine Learning algorithm. There are some problems with the dataset.

- All the images are of different size
- Some of the images are completely corrupted
- Some images don't have any faces
- Some of the ages are invalid
- The distribution between the gender is not equal(there are more male faces than female faces)
- Also, the meta information is in .mat format. Reading .mat files in python is a tedious process.


## Unzip dataset

After downloading, we've got folder wiki_crop. Unzip it to wiki_crop folder.


```python
cd /home/tranlysfw/age_gender_prediction/dataset/wiki
```


```python
unzip /home/tranlysfw/age_gender_prediction/dataset/wiki/wiki_crop.zip
```

## Create meta.csv

Running below script and we have file meta.csv. It split file_name to useful information


```python
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

imdb_mat = 'imdb_crop/imdb.mat'
wiki_mat = 'wiki_crop/wiki.mat'

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('imdb_crop/' + path[0])

for path in wiki_full_path:
    wiki_path.append('wiki_crop/' + path[0])

imdb_genders = []
wiki_genders = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
    else:
        imdb_genders.append('female')

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')

imdb_dob = []
wiki_dob = []

for file in imdb_path:
    temp = file.split('_')[3]
    temp = temp.split('-')
    if len(temp[1]) == 1:
        temp[1] = '0' + temp[1]
    if len(temp[2]) == 1:
        temp[2] = '0' + temp[2]

    if temp[1] == '00':
        temp[1] = '01'
    if temp[2] == '00':
        temp[2] = '01'
    
    imdb_dob.append('-'.join(temp))

for file in wiki_path:
    wiki_dob.append(file.split('_')[2])


imdb_age = []
wiki_age = []

for i in range(len(imdb_dob)):
    try:
        d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    imdb_age.append(diff)

for i in range(len(wiki_dob)):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    wiki_age.append(diff)

final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T
final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

meta = pd.concat((final_imdb_df, final_wiki_df))
meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']
meta = meta.drop(['face_score1', 'face_score2'], axis=1)
meta = meta.sample(frac=1)

meta.to_csv('meta.csv', index=False)

```

Reread meta.csv file and surveying some information


```python
df = pd.read_csv("/home/tranlysfw/age_gender_prediction/meta.csv")
df.head()
```


```python
file_path = df['path']
wiki_age = df['age']
wiki_gender = df['gender']
```

## Split images in wiki dataset to age and gender folders

Define the root of wiki dataset


```python
wiki_path = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/wiki")
wiki_male = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/wiki/gender/male")
wiki_female = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/wiki/gender/female")
wiki_age_path = pathlib.Path("/home/tranlysfw/age_gender_prediction//dataset/wiki/age")
```

After all, we use face_recognition lib to make sure it's getting exact human faces, not anything else. Then copying to suitable foldes


```python
i = 200000
alpha = 1.5 # ratio of margin from faces
face_count = 0
j = 200000
for elem in file_path[i:]:
    # print("i", i)
    # print("name",elem)
    dataset = elem.split("/")[0]
    # print("dataset", dataset)

    if dataset == "wiki_crop":
        # print("equal")
        abs_file_path = pathlib.Path(wiki_path, elem)
#         print("link", abs_file_path)
        img = cv2.imread(str(abs_file_path.absolute()))
        
        if img is not None:
            height, width, channel = img.shape
            face_locations = face_recognition.face_locations(img)

            for face in face_locations:
                face_count += 1
                # print("face detected")
                top, right, bottom, left = face  
                center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)
                border = int((right - left) * alpha)
              
                x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
                x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)
                
                if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
                    crop_face = img[y_up: y_up + border, x_left: x_left + border]
                    resized = cv2.resize(crop_face, (224, 224), interpolation=cv2.INTER_AREA)
                    
#                     print("gender", wiki_gender[i])
                    if wiki_gender[i] == "male":
                        path = pathlib.Path(wiki_male, elem.split("/")[2])
                        # print("save",path)
                        cv2.imwrite(str(path), resized)
                    elif wiki_gender[i] == "female":
                        path = pathlib.Path(wiki_female,elem.split("/")[2])
                        # print("save",path)
                        cv2.imwrite(str(path), resized)

                    age = wiki_age[i]
#                     print("age", age)
                    path_age = ""
                    if age <= 10:
                        path_age = pathlib.Path(wiki_age_path,"0", elem.split("/")[2])
                    elif age > 10 and age <= 20:
                        path_age= pathlib.Path(wiki_age_path, "1", elem.split("/")[2])
                    elif age > 20 and age <= 30:
                        path_age= pathlib.Path(wiki_age_path, "2", elem.split("/")[2])
                    elif age > 30 and age <= 40:
                        path_age= pathlib.Path(wiki_age_path, "3", elem.split("/")[2])
                    elif age > 40 and age <= 50:
                        path_age= pathlib.Path(wiki_age_path, "4", elem.split("/")[2])
                    elif age > 50 and age <= 60:
                        path_age= pathlib.Path(wiki_age_path, "5", elem.split("/")[2])
                    elif age > 60 and age <= 70:
                        path_age= pathlib.Path(wiki_age_path, "6", elem.split("/")[2])
                    elif age > 70:
                        path_age= pathlib.Path(wiki_age_path, "7", elem.split("/")[2])
                    # print("path_age",path_age)
                    cv2.imwrite(str(path_age), resized)
    i += 1
    if i  % 1000 == 0:
      print("loop",i)
      print("Face count", face_count)
         
```

<br>

## Test wiki dataset

Counting the number of files which are unzipped


```python
len(glob.glob("/home/tranlysfw/age_gender_prediction/dataset/wiki/age/*/*.jpg", recursive=True))
```


```python
len(glob.glob("/home/tranlysfw/age_gender_prediction/dataset/wiki/gender/*/*.jpg", recursive=True))
```

If there is something wrong, we should remove it before rerunning


```python
rm -R /home/tranlysfw/age_gender_prediction/dataset/wiki/age/7/*.jpg
```

Get some images from dataset


```python
img_test = list(glob.glob("/home/tranlysfw/age_gender_prediction/dataset/wiki/gender/female/*.jpg", recursive=True))
```


```python
test = cv2.imread(img[1061])
plt.imshow(test)
```

<br>

# IMDB

We process for IMDB dataset at exactly steps with wiki

## Unzip dataset

After downloading, we've got folder wiki_crop. We change directory to the same folder and run below script


```python
cd /home/tranlysfw/age_gender_prediction/dataset/imdb
```


```python
unzip /home/tranlysfw/age_gender_prediction/dataset/imdb_crop.zip
```

## Create meta.csv

Running below script and we have file meta.csv


```python
import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

imdb_mat = 'imdb_crop/imdb.mat'
wiki_mat = 'wiki_crop/wiki.mat'

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('imdb_crop/' + path[0])

for path in wiki_full_path:
    wiki_path.append('wiki_crop/' + path[0])

imdb_genders = []
wiki_genders = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append('male')
    else:
        imdb_genders.append('female')

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append('male')
    else:
        wiki_genders.append('female')

imdb_dob = []
wiki_dob = []

for file in imdb_path:
    temp = file.split('_')[3]
    temp = temp.split('-')
    if len(temp[1]) == 1:
        temp[1] = '0' + temp[1]
    if len(temp[2]) == 1:
        temp[2] = '0' + temp[2]

    if temp[1] == '00':
        temp[1] = '01'
    if temp[2] == '00':
        temp[2] = '01'
    
    imdb_dob.append('-'.join(temp))

for file in wiki_path:
    wiki_dob.append(file.split('_')[2])


imdb_age = []
wiki_age = []

for i in range(len(imdb_dob)):
    try:
        d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    imdb_age.append(diff)

for i in range(len(wiki_dob)):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        print(ex)
        diff = -1
    wiki_age.append(diff)

final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T
final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

meta = pd.concat((final_imdb_df, final_wiki_df))
meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']
meta = meta.drop(['face_score1', 'face_score2'], axis=1)
meta = meta.sample(frac=1)

meta.to_csv('meta.csv', index=False)

```

Reread meta.csv file and surveying some information


```python
df = pd.read_csv("/home/tranlysfw/age_gender_prediction/meta.csv")
df.head()
```


```python
file_path = np.array(df['path'])
imdb_age = np.array(df['age'])
imdb_gender = np.array(df['gender'])
```

## Split images in wiki dataset to age and gender folders

Define the root of wiki dataset


```python
imdb_path = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/imdb")
imdb_male = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/imdb/gender/male")
imdb_female = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/imdb/gender/female")
imdb_age_path = pathlib.Path("/home/tranlysfw/age_gender_prediction/dataset/imdb/age")
```

## Run script


```python
i = 0
j = 170000
alpha = 1.5
face_count = 0
start = time.time()
previous = time.time()

for elem in file_path[i:]:
#     print("i", i)
    # print("name",elem)
    dataset = elem.split("/")[0]
    

    if dataset == "imdb_crop":
        abs_file_path = pathlib.Path(imdb_path, elem)

#         print("link", abs_file_path)
        img = cv2.imread(str(abs_file_path.absolute()))

        if img is not None:
          
            height, width, channel = img.shape
            face_locations = face_recognition.face_locations(img)

            for face in face_locations:
                # print("face detected")
                face_count += 1
                top, right, bottom, left = face  
                center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)
                border = int((right - left) * alpha)
              
                x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
                x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)
                
                if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
                    crop_face = img[y_up: y_up + border, x_left: x_left + border]
                    resized = cv2.resize(crop_face, (224, 224), interpolation=cv2.INTER_AREA)
                
                    if imdb_gender[i] == "male":
                        path = pathlib.Path(imdb_male, elem.split("/")[2])
                        # print("save",path)
                        cv2.imwrite(str(path), resized)
                    elif imdb_gender[i] == "female":
                        path = pathlib.Path(imdb_female, elem.split("/")[2])
                        # print("save",path)
                        cv2.imwrite(str(path), resized)

                    path_age = ""
                  
                    age = imdb_age[i]
                    # print("age",age)
                    if age <= 10:
                        path_age= pathlib.Path(imdb_age_path, "0",elem.split("/")[2])
                    elif age > 10 and age <= 20:
                        path_age= pathlib.Path(imdb_age_path, "1",elem.split("/")[2])
                    elif age > 20 and age <= 30:
                        path_age= pathlib.Path(imdb_age_path, "2",elem.split("/")[2])
                    elif age > 30 and age <= 40:
                        path_age= pathlib.Path(imdb_age_path, "3",elem.split("/")[2])
                    elif age > 40 and age <= 50:
                        path_age= pathlib.Path(imdb_age_path, "4",elem.split("/")[2])
                    elif age > 50 and age <= 60:
                        path_age= pathlib.Path(imdb_age_path, "5",elem.split("/")[2])
                    elif age > 60 and age <= 70:
                        path_age= pathlib.Path(imdb_age_path, "6",elem.split("/")[2])
                    elif age > 70:
                        path_age= pathlib.Path(imdb_age_path, "7",elem.split("/")[2])

                    cv2.imwrite(str(path_age), resized)
      
    i += 1
    if i % 1000 == 0:
      print(i)
      print("Face count", face_count)
      print("epoch", time.time() - previous)
      previous = time.time()

end = time.time()
print("Time elapse: ", end - start)
      
```

<br>

## Test imdb dataset

Counting files after preprocessed


```python
len(glob.glob("/home/tranlysfw/age_gender_prediction/dataset/imdb/gender/*/*.jpg"))
```

Checking image from dataset


```python
img_test = list(glob.glob("/home/tranlysfw/age_gender_prediction/dataset/imdb/gender/female/*.jpg", recursive=True))
```


```python
test = cv2.imread(img_test[10])
plt.imshow(test)
```

<br>

# Asian

The file names include age only, information of gender will be pulled from other files.

## Split file names 


```python
project_path = pathlib.Path('D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience')
asian_folder_wild = pathlib.Path(os.path.join(project_path, "datasets//Asian//Asian//original images"))
asian_preprocess = pathlib.Path("D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience\\datasets\\Asian\\preprocess")
asian_male = pathlib.Path(os.path.join(asian_preprocess), "gender\\male\\")
asian_female = pathlib.Path(os.path.join(asian_preprocess), "gender\\female\\")
asian_age_path = pathlib.Path("D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience\\datasets\\Asian\\preprocess\\age")
```

Preprocess information file, we will have the all.csv file with two columns: file_name and gender


```python
df = pd.read_csv("D:\\01_PYTHON\\05_CoderS\\78_Ads_Targeted_Audience\\datasets\\Asian\\Asian\\image sets\\all.csv",)
```


```python
file_name = df["file"]
gender = df["gender"]
```

##  Crop and save preprocessed images


```python
img_paths = []
asian_age = []
asian_gender = []
i = 0
alpha = 1.5

for elem in file_name:
    abs_file_path = pathlib.Path(os.path.join(asian_folder_wild), elem)
    img = cv2.imread(str(abs_file_path.absolute()))
    height, width, channel = img.shape
    
    face_locations = face_recognition.face_locations(img)
    
    for face in face_locations:
        top, right, bottom, left = face  
        center_y, center_x = int((top + bottom) / 2), int((right + left) / 2)
        border = int((right - left) * alpha)
        
        x_right, y_up = int(center_x + border / 2), int(center_y - border / 2)
        x_left, y_down = int(center_x - border / 2), int(center_y + border / 2)
        
        if x_left > 0 and x_left + border < width and y_up > 0 and y_up + border < height:
            crop_face = img[y_up: y_up + border, x_left: x_left + border]
            resized = cv2.resize(crop_face, (224, 224), interpolation=cv2.INTER_AREA)
            
            filename_splited = elem.split("A")
            asian_age = int(filename_splited[1].split(".")[0])
            asian_gender = int(gender[i])
#             print(asian_gender)
            
            if asian_gender == 0:
                path = pathlib.Path(os.path.join(asian_female), elem)
#                 print(path)
                cv2.imwrite(str(path), resized)
            elif asian_gender == 1:
                path = pathlib.Path(os.path.join(asian_male), elem)
#                 print(path)
                cv2.imwrite(str(path), resized)
                
            path_age = ""
            if asian_age <= 10:
                path_age= pathlib.Path(os.path.join(asian_age_path), "0", elem.name)
            elif asian_age > 10 and asian_age <= 20:
                path_age= pathlib.Path(os.path.join(asian_age_path), "1", elem.name)
            elif asian_age > 20 and asian_age <= 30:
                path_age= pathlib.Path(os.path.join(asian_age_path), "2", elem.name)
            elif asian_age > 30 and asian_age <= 40:
                path_age= pathlib.Path(os.path.join(asian_age_path), "3", elem.name)
            elif asian_age > 40 and asian_age <= 50:
                path_age= pathlib.Path(os.path.join(asian_age_path), "4", elem.name)
            elif asian_age > 50 and asian_age <= 60:
                path_age= pathlib.Path(os.path.join(asian_age_path), "5", elem.name)
            elif asian_age > 60 and asian_age <= 70:
                path_age= pathlib.Path(os.path.join(asian_age_path), "6", elem.name)
            elif asian_age > 70:
                path_age= pathlib.Path(os.path.join(asian_age_path), "7", elem.name)
                
            cv2.imwrite(str(path_age), resized)
            
    i += 1
    if i % 200 == 0: 
        print(i)
```

<br>

# Reference

https://github.com/imdeepmind/processed-imdb-wiki-dataset
