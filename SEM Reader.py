#!/usr/bin/env python
# coding: utf-8

# In[174]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint
from scipy.stats import gaussian_kde 
import pytesseract
from PIL import Image


# In[ ]:


#This scripyt specifically for Scanning electron Microscopes. It takes their output, reads and prints the data in a copyable 
#format. We can paste into data files with this saaving hours of typing labor.


# In[6]:


#!pip install pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\21rgo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Open the .tif image using Pillow (PIL)
image_path = 'C:/Users/21rgo/OneDrive/Desktop/Rowan-9-26/6137/NWA 6137-c-3bulk.tif'  # Replace with the path to your .tif image
img = Image.open(image_path)

width, height = img.size


left = width // 2
top = 0
right = width
bottom = height
right_half = img.crop((left, top, right, bottom))

# Perform OCR on the right half
extracted_text = pytesseract.image_to_string(right_half, lang='eng')  # 'eng' for English
# Split the extracted text into lines
lines = extracted_text.splitlines()


word_index = 1   
print(extracted_text)
print('==============')
print('==============')

for line in lines:
    words = line.split()
    if word_index < len(words):
    # Print the specific word
        specific_word = words[word_index]
        print(specific_word)


# In[ ]:





# In[175]:


# Set  Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\21rgo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# open  .tif image using Pillow (PIL)
image_path = 'C:/Users/21rgo/OneDrive/Desktop/Rowan-9-26/6137/NWA 6137-a-3bulk.tif'
img = Image.open(image_path)

width, height = img.size


left = width  // 2
top = 0
right = width
bottom = height
right_half = img.crop((left, top, right, bottom))

custom_config = r' '
extracted_text = pytesseract.image_to_string(img, lang='eng', config=custom_config)

print(extracted_text)
print('=======')


