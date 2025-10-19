#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Executable X, y training data


datadir = "C:/Users/21rgo/OneDrive/Pictures"
catag = ["AuroraT","NoAuroraF"]

for category in catag:
    path = os.path.join(datadir, category)  # Note that it's "category", not "catag"
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))

    class_num = catag.index(category)

# pprint(img_array)
# print(img_array.shape)

img_size = 50

new_array = cv2.resize( img_array, (img_size, img_size, ))
#plt.imshow(new_array)
#plt.show()    
  

training_data = []

def create_training_data():
    for category in catag:
        path = os.path.join(datadir, category )
        class_num = catag.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])    
            
create_training_data()

# X = pickle.load(open("X.pickle","rb"))
# y = pickle.load(open("y.pickle","rb"))
from sklearn.utils import shuffle

import random
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1,img_size, img_size,3) 
X = np.array(X)
y = np.array(y)
X = X/255.0
X_o, y_o = shuffle(X, y, random_state=42)




# print(X)
# print(y)


# In[ ]:





# In[2]:


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
from sklearn.utils import shuffle

from keras.callbacks import EarlyStopping



# X = pickle.load(open("X.pickle","rb"))
# y = pickle.load(open("y.pickle","rb"))


# In[ ]:





# In[8]:


model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/0-conv-16-nodes-1-dense-1699247321 testingGB8.model")

# Define your categories (class labels)
catag = ["AuroraT", "NoAuroraF"]

def prepare(filepath):
    img_size = 50
    img_array = cv2.imread(filepath)
    
    if img_array is None:
        print(f"Error: Unable to read image at {filepath}")
        return 
    
  
    
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)

  

# Directory containing test images
#test_directory ="C:/Users/21rgo/Videos/CurrentFrames"   #CurrentFrames   Temp_INPUT
test_directory =  "C:/Users/21rgo/OneDrive/Pictures/Test_TruAurora"   #CurrentFrames   Temp_INPUT Test_TruAurora

def mTest(Ttest_directory, Ftest_directory):
    # List all image files in the directory
    Timage_files = [os.path.join(Ttest_directory, file) for file in os.listdir(Ttest_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    Ftot = 0 
    Ttot = 0
    T    = 0
    F    = 0
    for image_path in Timage_files:

        # Make predictions for each image
        prepared_image = prepare(image_path)
        predictions = model.predict(prepared_image)

        # Get the predicted class label
        predicted_class_index = int(predictions[0][0])
        predicted_class = catag[predicted_class_index]

        #fraction of correctly guessing true

        if predicted_class == 'NoAuroraF':
            Ttot = Ttot + 1
        if predicted_class == 'AuroraT':
            Ttot = Ttot + 1
            T = T + 1
        
        if F==0:
                pass    
        fracT= T / Ttot
    

    Fimage_files = [os.path.join(Ftest_directory, file) for file in os.listdir(Ftest_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in Fimage_files:
        
        if prepared_image is not None:
            # Make predictions for each image
            prepared_image = prepare(image_path)

            predictions = model.predict(prepared_image)
    

            # Get the predicted class label
            predicted_class_index = int(predictions[0][0])
            predicted_class = catag[predicted_class_index]

            #fraction of correctly guessing true

            if predicted_class == 'NoAuroraF':
                Ftot = Ftot + 1
                F = F + 1

            if predicted_class == 'AuroraT':
                Ftot = Ftot + 1

            if F==0:
                pass
            fracF = F / Ftot

    print(fracF, "percent score on finding  false Aurora")
    print(fracT, "percent score on finding  true Aurora")


    Score = (fracT + fracF )/2
    print('==========Overall:',Score)


# In[6]:


import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Assuming X_o, y_o are your original features and labels
# Assuming X_n, y_n are your new features and labels

# Create a weighted sampling strategy
weights = [0.9] * len(X_o) + [0.1] * len(X_n)  # Assign higher weight to old data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Use numpy.concatenate for concatenation
X_combined = np.concatenate((X_o, X_n), axis=0)
y_combined = np.concatenate((y_o, y_n), axis=0)

# Use the weighted sampling strategy for splitting data
for train_index, test_index in sss.split(X_combined, y_combined, weights):
    X_train, X_test = X_combined[train_index], X_combined[test_index]
    y_train, y_test = y_combined[train_index], y_combined[test_index]

# Now, you can use X_train, y_train for training your model
# and X_test, y_test for evaluation



# In[91]:


optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)
batch_size = 6
NAME = "V2modelGB8--15".format( int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Load your pre-trained model
model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/Best Models/0-conv-16-nodes-1-dense-1699247321 testingGB8.model")


# Compile the model with the same configuration
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Define your new hyperparameters
new_epochs = 22  # Adjust this to the desired number of additional epochs

# Train the model for additional epochs
model.fit(X_train, y_train, batch_size=batch_size, epochs=new_epochs, validation_split=0.1, callbacks=[tensorboard])
model.save('logs/{}.model'.format(NAME))


# In[18]:


model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/Alto259GB8--Edits2.model")

#loss, accuracy = model.evaluate(X, y)


# In[94]:


model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/V2modelGB8--15.model")
#model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/0-conv-16-nodes-1-dense-Alto259GB8.model - Copy")
#new
Ttest_directory =  "C:/Users/21rgo/OneDrive/Pictures/TrueTrainDat"      #Test_TruAurora
Ftest_directory =  "C:/Users/21rgo/OneDrive/Pictures/FalseNewTrainigData copy"   #NoAuroraF   Temp_INPUT  Test_FalsAurora 

#old test
# Ttest_directory =  "C:/Users/21rgo/OneDrive/Pictures/Test_TruAurora"      #Test_TruAurora
# Ftest_directory =  "C:/Users/21rgo/OneDrive/Pictures/Test_FalsAurora"   

mTest(Ttest_directory, Ftest_directory)




# In[8]:


obtains aurora that werent detected.
# model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/0-conv-16-nodes-1-dense-1699247321 testingGB8.model")

# # Define your categories (class labels)
# catag = ["AuroraT", "NoAuroraF"]

# def prepare(filepath):
#     img_size = 50
#     img_array = cv2.imread(filepath)
#     new_array = cv2.resize(img_array, (img_size, img_size))
#     return new_array.reshape(-1, img_size, img_size, 3)

  

# # Directory containing test images
# #test_directory ="C:/Users/21rgo/Videos/CurrentFrames"   #CurrentFrames   Temp_INPUT
# test_directory =  "C:/Users/21rgo/OneDrive/Pictures/Test_TruAurora"   #CurrentFrames   Temp_INPUT Test_TruAurora

# def TruFailedTest(Ttest_directory, Ftest_directory):
#     # List all image files in the directory
#     TrueTrainDat = 'C:/Users/21rgo/OneDrive/Pictures/TrueTrainDat'
#     Timage_files = [os.path.join(Ttest_directory, file) for file in os.listdir(Ttest_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     Ftot = 0 
#     Ttot = 0
#     T    = 0
#     F    = 0
#     for image_path in Timage_files:

#         # Make predictions for each image
#         prepared_image = prepare(image_path)
#         predictions = model.predict(prepared_image)

#         # Get the predicted class label
#         predicted_class_index = int(predictions[0][0])
#         predicted_class = catag[predicted_class_index]

#         #fraction of correctly guessing true

#         if predicted_class == 'NoAuroraF':
#             Ttot = Ttot + 1
#             # Extract the image file name from the full path
#             image_file_name = os.path.basename(image_path)
#               # Define the new path where you want to move the image
#             new_path = os.path.join(TrueTrainDat, image_file_name)
#                          # Move the image to the FalseNewData directory
#             shutil.copy(image_path, new_path)             
              
            
#         if predicted_class == 'AuroraT':
#             Ttot = Ttot + 1
#             T = T + 1

#         fracT= T / Ttot
    

#     Fimage_files = [os.path.join(Ftest_directory, file) for file in os.listdir(Ftest_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     for image_path in Fimage_files:

#         # Make predictions for each image
#         prepared_image = prepare(image_path)
#         predictions = model.predict(prepared_image)

#         # Get the predicted class label
#         predicted_class_index = int(predictions[0][0])
#         predicted_class = catag[predicted_class_index]

#         #fraction of correctly guessing true

#         if predicted_class == 'NoAuroraF':
#             Ftot = Ftot + 1
#             F = F + 1

#         if predicted_class == 'AuroraT':
#             Ftot = Ftot + 1


#         fracF = F / Ftot
        
#     print(fracF, "percent score on finding  false Aurora")
#     print(fracT, "percent score on finding  true Aurora")


#     Score = (fracT + fracF )/2
#     print('==========Overall:',Score)


# In[ ]:





# In[ ]:





# In[ ]:




