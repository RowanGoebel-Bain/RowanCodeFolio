#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import shutil


# In[ ]:


#MULTIVIDEO FILE UPLOAD    
    #CHECKLIST
        #output dir
        #name the frames (version)
        #frame count
hold
        
source_directory = "C:/Users/21rgo/Videos/Temp_INPUT"    #empites source into destination when upload is complete
destination_directory = "C:/Users/21rgo/Videos/NONaurora_Input"

# List all files in the folder
video_files = [file for file in os.listdir(source_directory) if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]

version = 0

# Iterate through video files
for video_file in video_files:
    video_path = os.path.join(source_directory, video_file).replace("\\", "/")

    print("Processing video:", video_path)

    output_directory = "C:/Users/21rgo/OneDrive/Pictures/NoAuroraF"  #   AuroraT    NoAuroraF
    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    version += 1

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Capture every fortieth frame
        if frame_count % 80 == 0:
            frame_filename = os.path.join(output_directory, f'V_{version}__Auroraframe_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Saved frame {frame_count}')

    cap.release()
    print(f'Frames saved to: {output_directory}')

    # Moving directory
    source_path = os.path.join(source_directory, video_file)
    destination_path = os.path.join(destination_directory, video_file)

    # Check if the source file exists and move it to the destination
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f"Moved: {video_file} to {destination_directory}")
    else:
        print(f"Source file not found: {video_file}")

print("Done moving video files.")


# In[3]:


#SINGLE VIDEO UPLOAD
   #CHECKLIST
       #output dir
       #name the frames (version /location)
       #frame count
       

       
       
output_directory = "C:/Users/21rgo/Videos/Temp_INPUT"  #   AuroraT   NoAuroraF Test_Aurora
os.makedirs(output_directory, exist_ok=True)

vid_upload = "C:/Users/21rgo/Videos/Aurora_Input/2023_03_24_03_04_00_000_011041.mp4"
# Open the video file
cap = cv2.VideoCapture(vid_upload)

frame_count = 0
#version += 1
version = 0    # manually set

while cap.isOpened():
   ret, frame = cap.read()

   if not ret:
       break

   frame_count += 1

# Capture every fortyth frame
   if frame_count % 111 == 0:


       frame_filename = os.path.join(output_directory, f'TrueTesting_V_{version}__Auroraframe_{frame_count}.jpg')
       cv2.imwrite(frame_filename, frame)
       print(f'Saved frame {frame_count}')

cap.release()
# cv2.destroyAllWindows() (not needed)
print(f'Frames saved to: {output_directory}')


# In[ ]:


#DELETE A SET OF ACCIDENTAL IMAGES
hold

# Path to the folder containing the images
folder_path = 'C:/Users/21rgo/OneDrive/Pictures/NoAuroraF'

# List all files in the folder
files = os.listdir(folder_path)

# Filter the list to include only image files (you can adjust the file extensions as needed)
image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# Sort the image files by modification time in descending order (most recent first)
image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

# Delete the 150 most recent images
images_to_delete = image_files[:155]

for image in images_to_delete:
    image_path = os.path.join(folder_path, image)
    os.remove(image_path)
    print(f"Deleted: {image_path}")


# In[7]:


#makes a test file out of the true training data
import os
import random
import shutil
hold
# Source folder with photos
source_folder = "C:/Users/21rgo/OneDrive/Pictures/AuroraT"

# Destination folder to copy the selected photos
destination_folder = "C:/Users/21rgo/OneDrive/Pictures/Test_TruAurora"

# Number of photos to select and copy
num_photos_to_copy = 50

# Get a list of all photo files in the source folder
photo_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Randomly select num_photos_to_copy photos
selected_photos = random.sample(photo_files, num_photos_to_copy)

# Copy the selected photos to the destination folder
for photo in selected_photos:
    source_path = os.path.join(source_folder, photo)
    destination_path = os.path.join(destination_folder, photo)
    shutil.copy2(source_path, destination_path)

print(f"Selected and copied {num_photos_to_copy} photos to the destination folder.")


# In[ ]:




