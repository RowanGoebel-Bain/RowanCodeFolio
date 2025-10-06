#!/usr/bin/env python
# coding: utf-8

# In[46]:


#Executable X, y
# X = pickle.load(open("X.pickle","rb"))
# y = pickle.load(open("y.pickle","rb"))

import random
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size,3) 
X = np.array(X)
y = np.array(y)
X = X/255.0
X, y = shuffle(X, y, random_state=42)




print(X)
print(y)


# In[114]:


#Class for Green Channel
import tensorflow as tf
from tensorflow.keras.layers import Layer

class RGBChannelLayer(Layer):
    def __init__(self):
        super(RGBChannelLayer, self).__init__()

    def build(self, input_shape):
        super(RGBChannelLayer, self).build(input_shape)

    def call(self, inputs):
        # Separate the color channels (assuming the input shape is NHWC)
        red_channel = inputs[:, :, :, 0:1]
        green_channel = inputs[:, :, :, 1:2]
        blue_channel = inputs[:, :, :, 2:3]

        # Combine the color channels back into a single RGB tensor
        rgb_image = tf.concat([red_channel, green_channel, blue_channel], axis=-1)

        return rgb_image
    
    
class GreenChannelLayer(Layer):
    def __init__(self, **kwargs):
        super(GreenChannelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GreenChannelLayer, self).build(input_shape)

    def call(self, inputs):
        # Extract the green channel from the input image
        green_channel = inputs[:, :, :, 1:2]  # Green channel is at index 1

        return green_channel

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


# In[43]:


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



X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))


# In[2]:


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
  

 


# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint("model_weights.h5", save_best_only=True)

# Load your model
model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/trials.keras")

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use the callback in the fit method
model.fit(training_data, catag, epochs=20, callbacks=[checkpoint_callback])


# In[3]:


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


# In[163]:


with open('training_data.pickle', 'wb') as file:
    pickle.dump(training_data, file)


# In[32]:


print(len(training_data))
import random
random.shuffle(training_data)

from tensorflow.keras.optimizers import SGD




# In[ ]:





# In[17]:


# Define the path where you want to save the model
#NAME = "{}-conc-{}-nodes-{}-dense-{}".format(dense_layer,layer_size,conv_layer, int(time.time()))
random.shuffle(training_data)

batch_size = 64
epochs = 1
learning_rate = 0.001

#optimizer = Adam(learning_rate=learning_rate) # Create an optimizer with your desired learning rate
#optimizer = SGD(learning_rate= learning_rate)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Stop after 5 epochs of increasing validation loss


dense_layers = [0,1]
layer_sizes = [32,64,128]
conv_layers = [1,3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(dense_layer,layer_size,conv_layer, int(time.time()))
            print(NAME)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


            model = Sequential()
            
            model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]) )
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3)))
                model.add(BatchNormalization())
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

            model.add(Flatten())
            for l in range(dense_layer ):
                model.add(Dense(layer_size))
                model.add(BatchNormalization())
                model.add(Activation("relu"),kernel_regularizer=l2(0.01))
                model.add(Dropout(0.5))

            #model.add(GlobalAveragePooling2D())  # Global Average Pooling layer

            #model.add(Dense(64, activation='relu'))  # Example fully connected layer
            #model.add(Activation("relu"))

            model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification


            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
            model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard  ])
            model.save('logs/{}.model'.format(NAME))



# In[ ]:





# In[139]:


#specific training model
from tensorflow.keras.optimizers import SGD

dense_layer = 0
layer_size = 64
conv_layer = 1
# Define the path where you want to save the model
#NAME = "{}-conc-{}-nodes-{}-dense-{}".format(dense_layer,layer_size,conv_layer, int(time.time()))
batch_size = 64
epochs = 8
learning_rate = 0.001

#optimizer = Adam(learning_rate=learning_rate) # Create an optimizer with your desired learning rate
#optimizer = SGD(learning_rate= learning_rate)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Stop after 5 epochs of increasing validation loss

NAME = "{}-conv-{}-nodes-{}-dense-{}gb9".format(dense_layer,layer_size,conv_layer, int(time.time()))
print(NAME)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()
model.add(GreenChannelLayer(input_shape=(img_size, img_size, 3)))

model.add(RGBChannelLayer())

#model.add(GreenChannelLayer(input_shape=(img_size, img_size, 3)))



model.add(Conv2D(layer_size,(3,3), input_shape = X.shape[1:]) )
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



for l in range(conv_layer-1):
    model.add(Conv2D(layer_size,(3,3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2 )))
    model.add(Dropout(0.25))



model.add(Flatten())
for l in range(dense_layer ):
    model.add(Dense(layer_size))
    model.add(BatchNormalization())
    model.add(Activation("relu"),kernel_regularizer=l2(0.01))
    model.add(Dropout(0.5))
    


#model.add(GlobalAveragePooling2D())  # Global Average Pooling layer

model.add(Dense(64, activation='relu'))  # Example fully connected layer
model.add(Activation("relu"))

model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification


model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard  ])

model.save('logs/{}.model'.format(NAME))




# In[37]:


#print(X)
print(y)


# In[164]:


model = tf.keras.models.load_model("C:/Users/21rgo/The Plus 1/MMGM Work/logs/0-conv-16-nodes-1-dense-1699247321 testingGB8.model")

loss, accuracy = model.evaluate(X, y)


# In[162]:


def prepare(filepath):
    img_size = 50
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3)

test_directory =  "C:/Users/21rgo/OneDrive/Pictures/Test_TruAurora"   #CurrentFrames   Temp_INPUT Test_TruAurora
#test_directory = "C:/Users/21rgo/Videos/CurrentFrames"
# List all image files in the directory
image_files = [os.path.join(test_directory, file) for file in os.listdir(test_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]



tot = 0
T   = 0
for image_path in image_files:

    # Make predictions for each image
    prepared_image = prepare(image_path)
    predictions = model.predict(prepared_image)
   
    # Get the predicted class label
    predicted_class_index = int(predictions[0][0])
    predicted_class = catag[predicted_class_index]
    
    #fraction of correctly guessing true

    if predicted_class == 'NoAuroraF':
        tot = tot + 1
    if predicted_class == 'AuroraT':
        tot = tot + 1
        T = T + 1


    print(f"Image: {image_path}, Predicted Class: {predicted_class}")
    frac = T / tot
print(frac, "percent score on finding Aurora")


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


# optimizer 
# learning rate
# number of layers 
# nodes per layer
# dense layers
# unites per alyer
# activation units
# kernel size
# stride
# decay
# decay rate


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




