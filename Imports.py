#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This script takes in the training data and allows for some hyperperameter tuning. 
import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.callbacks import TensorBoard
import random
import smtplib

# Define constants
IMG_SIZE = 50
DATADIR = "C:/Users/21rgo/OneDrive/Pictures"
CATEGORIES = ["AuroraT", "NoAuroraF"]
NUM_EPOCHS = 10

# Function to create training data
def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = new_array / 255.0  # Normalize image data
                training_data.append([new_array, class_num])
            except Exception as e:
                # Handle exceptions when reading images
                pass
    return training_data

# Shuffle the training data
training_data = create_training_data()
random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Initialize TensorBoard
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define hyperparameter combinations
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# Model building loop
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2))
                      
           for l in range(conv_layer -1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            
            model.add(Flatten())
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
            
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer="adam", metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs=NUM_EPOCHS, validation_split=0.1, callbacks=[tensorboard])


# In[7]:


def send_notification(subject, message, to_email):
    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'pleverone@mainemineralmuseum.org'
    smtp_password = '****'

    # Compose the email
    from_email = smtp_username
    email_subject = subject
    email_body = message
    email_text = f"Subject: {email_subject}\n\n{email_body}"

    # Connect to the SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)

    # Send the email
    server.sendmail(from_email, to_email, email_text)

    # Quit the server
    server.quit()

# Example usage
if 1 == 1:
    send_notification("Alert", "The Aurora is on!", "rowan.m.goebel-bain.25@dartmouth.edu")

