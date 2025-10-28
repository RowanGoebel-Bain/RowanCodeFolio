#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# this code is intended for X-ray Fluorescnece microscope. It reads and prints its output data to save typing labor and time
#With this tool, we quickly paste columns of data into our tables.


# In[6]:


# Create dictionaries to store data for each element
import pytesseract
from PIL import Image       #designed for XRF readings
MgO = 'MgO'
Al2O3 = 'Al203'
SiO2 = 'SiO2'
Ca = 'Ca'
Fe = 'Fe'
S = 'S'
P = 'P'
Cr = 'Cr'
Ti = 'Ti'
K20 = 'K20'  
Mn = 'Mn' 
Ni = 'Ni' 
Co ='Co'
Cu = 'Cu'
Zn = 'Zn'
V = 'v'
Cd = 'Cd'
Pb = 'Pb'
Sr = 'Sr'
Ta = 'Ta'
Sb = 'Sb'
Zr = 'Zr'
Ga = 'Ga'
Nb = 'Nb'
Se = 'Se'
Ba = 'Ba'
#correct spellings: (remane for package derived errors)
AI2O3 = 'AI203'
Vv = 'Vv'

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\21rgo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Open the .tif image using Pillow (PIL)
image_path = 'C:/Users/21rgo/Downloads/6137/6137-XRF-5-a.jpg'
#image2_path = 'C:/Users/21rgo/Downloads/6137/6137-XRF-5-b.jpg'

img = Image.open(image_path)
#img2 = Image.open(image2_path)

# Perform OCR using Tesseract with the custom configuration
extracted_text = pytesseract.image_to_string(img, lang='eng')
# Split the extracted text into lines
lines = extracted_text.splitlines()

Elements = [MgO, Al2O3, SiO2, Ca, Fe, S, P, Cr, Ti, K20, Mn, Ni, Co, Cu, Zn, V, Cd, Pb, Sr, Ta, Sb, Zr, Ga, Nb, Se, Ba]  
element_data = {}
for element in Elements:
    element_data[element] = {'comp': [0], '2sig': [0]}  # Initialize with zeroes
    
for line in lines:
    words = line.split()
    if AI2O3 in words:
        Al_index = words.index(AI2O3)
        words[Al_index] = Al2O3
        extracted_text = extracted_text.replace(AI2O3, Al2O3)      # Replace the misspelling with the correct word
        
    if Vv in words:
        V_index = words.index(Vv)
        words[V_index] = V
        extracted_text = extracted_text.replace(Vv, V)
        
    for element in Elements:
        if element in words:
            word_index = words.index(element)
            if word_index + 2 < len(words):
                # Convert the string to a numeric value (float)
                try:
                    numeric_value = float(words[word_index + 1])
                except ValueError:
                    print(f"Warning: Unable to convert '{words[word_index + 1]}' to numeric value for element '{element}'")
                    continue  # Skip this element and move to the next one

                # Create a dictionary for the element if it doesn't exist
                if element not in element_data:
                    element_data[element] = {'comp': [], '2sig': []}
                    
                if element_data[element]['comp'][0] == 0:
                    element_data[element]['comp'] = []  # Remove the initial zero
                    element_data[element]['2sig'] = []
                # Append the values to the respective lists in the element's dictionary
                    element_data[element]['comp'].append(numeric_value)
                    element_data[element]['2sig'].append(words[word_index + 2])
                
# Now, element_data contains the data for each element, including composition and 2sigma values

#print(element_data)
#print(extracted_text)
# Iterate through the Elements list and print composition values in order
print('==Percent Comp:')
for element in Elements:
    if element in element_data:
        comp_values = element_data[element]['comp']
        comp_values_str = ', '.join(map(str, comp_values))  # Convert list to a comma-separated string
        sig_values = element_data[element]['2sig']
        sig_values_str = ', '.join(map(str, sig_values))  # Convert list to a comma-separated string
        
        print( comp_values_str)
print('==Sigma Values:')        
for element in Elements:
    if element in element_data:
        comp_values = element_data[element]['comp']
        comp_values_str = ', '.join(map(str, comp_values))  # Convert list to a comma-separated string
        sig_values = element_data[element]['2sig']
        sig_values_str = ', '.join(map(str, sig_values))  # Convert list to a comma-separated string        
        
        print( sig_values_str)
        
        
  #=============Mineral Ratio Code==========

            #Here we aim to read in elemental comps from our dictionary and calculate mineral ratios
    #Note: There will be an error when reading XRF page 2. This won't affect the above data
if 'MgO' in element_data:
    MgO_minerals = element_data['MgO']['comp']
    MgO_minerals = [float(value) for value in MgO_minerals] 
    MgO_minerals = MgO_minerals[0]

    
if 'Ca' in element_data:
    Ca_minerals = element_data['Ca']['comp']
    Ca_minerals = [float(value) for value in Ca_minerals]
    Ca_minerals = Ca_minerals[0]

if 'Fe' in element_data:
    Fe_minerals = element_data['Fe']['comp']
    Fe_minerals = [float(value) for value in Fe_minerals]  
    Fe_minerals = Fe_minerals[0]
    
# Fe_minerals = 13.70247143   #Be sure to insert XRF given averages BEFORE you have done stochiometric oxide conversion
# MgO_minerals = 20.98825714  
# Ca_minerals = 1.209357143
Na_minerals = 0.558    # enter this manually if you have it from SEM.

Fe2O3 = ((Fe_minerals*1.4297)/159.7) #oxide conversion by molecular weight
MgO = ((MgO_minerals)/40.31)
CaO = ((Ca_minerals*1.3992)/56.08)
Na2O = (((Na_minerals)/61.98)) 
      
Pyrox = Fe2O3 + MgO + CaO 
Olivine = Fe2O3 + MgO
Plag = CaO + Na2O

Fe_Pyrox = round((Fe2O3/Pyrox),6)
Mg_Pyrox = round((MgO/Pyrox),6)
Ca_Pyrox = round((CaO/Pyrox),6)
Fe_Olivine = round((Fe2O3/Olivine),6)
Mg_Olivine = round((MgO/Olivine),6)
Ca_Plag = round((CaO/Plag),6)
Na_Plag = round((Na2O/Plag),6)

print("Fe pyrx percent abundance",Fe_Pyrox*100)
print("Mg pyrx percent abundance",Mg_Pyrox*100)
print("Ca pyrx percent abundance",Ca_Pyrox*100)
print("===BREAK===")
print("Fe Olivine percent abundance",Fe_Olivine*100)
print("Mg Olivine percent abundance",Mg_Olivine*100)
print("===BREAK===")
print("Ca Plag percent abundance",Ca_Plag*100)
print("Na Plag percent abundance",Na_Plag*100)
print("===BREAK===")

print(Fe_Pyrox*100)
print(Mg_Pyrox*100)
print(Ca_Pyrox*100)
print(Fe_Olivine*100)
print(Mg_Olivine*100)
print(Ca_Plag*100)
print(Na_Plag*100)
print("This column can be copy/pasted into sheets")



# In[ ]:


#Once you have averages across samples:

Fe = 13.70247143  
Ca = 1.209357143
S =1.221
P =0.124242
Ti =0.04665714286
Cr =0.321
Mn =0.2502714286
Ni =0.3163285714
V =0.01752857143
Cu =0.0028
Zn =0.0047
print("Percent weight Fe2O3:",round((Fe*1.4297),4))
print("Percent weight CaO:",round((Ca*1.3992),4))
print("Percent weight SO3:",round((S*2.4972),4))
print("Percent weight P2O5:",round((P*2.2916),4))
print("Percent weight TiO2:",round((Ti*1.6681),4))
print("Percent weight Cr2O3:",round((Cr*1.4615),4))
print("Percent weight MnO:",round((Mn*1.2912),4))
print("Percent weight NiO:",round((Ni*1.2725),4))
print("Percent weight V2O5:",round((V*1.7852),4))
print("Percent weight CuO:",round((Cu*1.2518),4))
print("Percent weight ZnO:",round((Zn*1.3508),4))
print("These values have been converted to percent")

#feel free to copy/paste/replace If you want to calculate new elements

