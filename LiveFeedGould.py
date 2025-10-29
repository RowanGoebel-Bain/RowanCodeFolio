#!/usr/bin/env python
# coding: utf-8

# In[87]:


model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/Best Models/0-conv-16-nodes-1-dense-1699247321 testingGB8.model")


# In[ ]:


#This script loads in the pre trained model and makes predictions based on the live feed several times per minute.


# In[129]:


#Executable
#!pip install keyboard
import schedule
import keyboard
import cv2
import os
import time
import tensorflow as tf
import pysftp


def execute_tasks():
    download_most_recent_videos()
    LiveExtraction()
    predict()
    
    directory = r"C:/Users/21rgo/Videos/LIVE_FOOD"
    file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)])

    if file_count > 30:
        ClearFeed()

# Schedule the tasks to run every 15 minutes
schedule.every(1).minutes.do(execute_tasks)

# Keep the script running to execute tasks on schedule
while True:
    schedule.run_pending()
    time.sleep(1)  # Sleep to avoid high CPU usage


# In[115]:


remote_directory = "/mnt/ams2/HD"

def download_most_recent_videos():
        # Configure SFTP connection options
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    localdir = 'C:/Users/21rgo/Videos/LIVE_FOOD'
    # SFTP connection settings
    sftp_host = 'mmgm-ams.gouldacademy.org'
    sftp_username = 'ams'
    sftp_password = 'xrp23q'
    remote_directory = '/mnt/ams2/HD'  

    # Establish an SFTP connection
    with pysftp.Connection(sftp_host, username=sftp_username, password=sftp_password, cnopts=cnopts) as sftp:
        # List files in the remote directory
        remote_file_info = sftp.execute(f"ls -lt {remote_directory}")
         # Extract file names, sort by modification time (newest first), and select the top 7 files
        #remote_files = [f for f in remote_files if sftp.exists(remote_directory + '/' + f)]
        remote_files = [line[-35:] for line in remote_file_info if line.strip()]
        remote_files = [line[-35:].decode('utf-8') for line in remote_file_info if line.strip()]

        remote_files = [f[:-1] for f in remote_files]
        del remote_files[0]
        for file in remote_files[:14]:
            #print(file)
            if file.endswith("1.mp4") or file.endswith("2.mp4"):
                remote_path = f"{remote_directory}/{file}"
                local_path = os.path.join(localdir, file)

                # Download the file using pysftp
                sftp.get(remote_path, local_path)

    
#download_most_recent_videos()


# In[118]:


# extract frames from live video. This is done for additional training data
def LiveExtraction():
    source_directory = "C:/Users/21rgo/Videos/LIVE_FOOD" 
    test_directory = "C:/Users/21rgo/Videos/CurrentFrames"
    video_files = [file for file in os.listdir(source_directory) if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    iteration = 0
    for video_file in video_files:
        video_path = os.path.join(source_directory, video_file).replace("\\", "/")
        print("Processing video:", video_path)

        frame_count = 0
        iteration += 1
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame_count += 1

            # Capture every 250 frames
            if frame_count % 240 == 0:
                frame_filename = os.path.join(test_directory, f'VidFileIterate{iteration}_CurrentFrame_{frame_count}.jpg')
                frame_filename = frame_filename.replace("\\", "/")

                cv2.imwrite(frame_filename, frame)
                #print(f'Saved frame {frame_count}')

        cap.release()
        if iteration >= 2:
            iteration = 0


#LiveExtraction()


# In[112]:


def predict():
    # Define your categories (class labels)
    catag = ["AuroraT", "NoAuroraF"]

    def prepare(filepath):
        img_size = 50
        img_array = cv2.imread(filepath)
        new_array = cv2.resize(img_array, (img_size, img_size))
        return new_array.reshape(-1, img_size, img_size, 3)

    # Directory containing test images
    test_directory ="C:/Users/21rgo/Videos/CurrentFrames"

    # List all image files in the directory
    image_files = [os.path.join(test_directory, file) for file in os.listdir(test_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in image_files:
        # Make predictions for each image
        prepared_image = prepare(image_path)
        predictions = model.predict(prepared_image)

        # Get the predicted class label
        predicted_class_index = int(predictions[0][0])
        predicted_class = catag[predicted_class_index]
        
        if predicted_class == 'AuroraT':
            print('The Redcoats are coming!')
            

        #print(f"Image: {image_path}, Predicted Class: {predicted_class}")
        #print(predictions)


# In[127]:


def ClearFeed()  : 
    directory = r"C:/Users/21rgo/Videos/LIVE_FOOD"

    files = os.listdir(directory)

    # Sort the files by modification time (newest first)
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # Define how many recent files to keep
    num_files_to_keep = 4

    # Loop through the files and remove files beyond the specified number to keep
    for file in files[num_files_to_keep:]:
        file_path = os.path.join(directory, file)
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except PermissionError:
            print(f"Skipped: {file_path} (in use by another process)")

        
        
#ClearFeed()

