#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

bigpath = r"C:\Users\21rgo\Downloads\dr9_v2.0_MRS_stellar.csv.gz"
df = pd.read_csv(bigpath)
print(df.columns)


# In[9]:


df['ca_fe'] = pd.to_numeric(df['ca_fe'], errors='coerce')
df['feh_cnn'] = pd.to_numeric(df['feh_cnn'], errors='coerce')


dfca = pd.to_numeric(df['ca_fe'], errors='coerce')
dfca = dfca[dfca.notna()].astype(float)

dffeh = pd.to_numeric(df['feh_cnn'], errors='coerce')
dffeh = dffeh[dffeh.notna()].astype(float)



df999 = df.dropna(subset=['feh_cnn','ca_fe'])

print(len(df999))
print(len(dffeh))
print(len(dfca))


# In[11]:


df['feh_cnn'] = pd.to_numeric(df['feh_cnn'], errors='coerce')
df['ni_fe'] = pd.to_numeric(df['ni_fe'], errors='coerce')
df['n_fe'] = pd.to_numeric(df['n_fe'], errors='coerce')
df['ca_fe'] = pd.to_numeric(df['ca_fe'], errors='coerce')
df['c_fe'] = pd.to_numeric(df['c_fe'], errors='coerce')
df['mg_fe'] = pd.to_numeric(df['mg_fe'], errors='coerce')


df['feh_cnn'] = df['feh_cnn'].replace(-9999.00, pd.NA)
df['ni_fe'] = df['ni_fe'].replace(-9999.00, pd.NA)
df['n_fe'] = df['n_fe'].replace(-9999.00, pd.NA)
df['ca_fe'] = df['ca_fe'].replace(-9999.00, pd.NA)
df['c_fe'] = df['c_fe'].replace(-9999.00, pd.NA)
df['mg_fe'] = df['mg_fe'].replace(-9999.00, pd.NA) #data cleaning


columns_to_clean = ['feh_cnn', 'ni_fe', 'n_fe', 'ca_fe', 'c_fe', 'mg_fe']
df[columns_to_clean] = df[columns_to_clean].apply(pd.to_numeric, errors='coerce').replace(-9999.00, pd.NA)

# Remove rows with missing values
df = df.dropna(subset=['feh_cnn', 'ni_fe','n_fe','ca_fe','c_fe','mg_fe'])

dffeh =df['feh_cnn']
dfc = df[(df['c_fe'] < -0.26) & (df['feh_cnn'] > -0.6)]
dfn = df[df['n_fe'] > 0.5]
dfni = df[df['ni_fe']< -.175]
dfca = df[df['ca_fe'] <-0.35]
dfmg = df[df['mg_fe'] < -0.15]

print(len(dffeh))
print(len(dfc))
print(len(dfn))
print(len(dfni))
print(len(dfca))
print(len(dfmg))


# In[12]:


total = 1339377
c = 7396/total
n = 3469/total
ni = 3449/total
ca = 7012/total
mg = 3583/total

print("percent c:", c)
print("percent n:", n)
print("percent ni:", ni)
print("percent ca:", ca)
print("percent mg:", mg)


# In[13]:


ni_c = len(dfc.index.intersection(dfni.index))
expect_ni_c = c*ni*total
#formula and print all the values
print("stars in ni and carbon:", ni_c,"expected _ni_c value:", expect_ni_c )
ni_c_correlation = ((ni_c/expect_ni_c)*100)  -100
print('percent change abunance from random distribution:', ni_c_correlation, '%') 

n_ni = len(dfni.index.intersection(dfn.index))
expect_n_ni = ni*n*total
print("stars in Nitrogen and nickel:", n_ni,"expected _n_ni value:", expect_n_ni )
n_ni_correlation = ((n_ni/expect_n_ni)*100)  -100
print('percent change abunance from random distribution:', n_ni_correlation, '%')

ni_mg = len(dfmg.index.intersection(dfni.index))
expect_ni_mg = mg*ni*total
print("stars in ni and mg:", ni_mg,"expected _ni_mg value:", expect_ni_mg )
ni_mg_correlation = ((ni_mg/expect_ni_mg)*100)  -100
print('percent change abunance from random distribution:', ni_mg_correlation, '%')

ni_c = len(dfc.index.intersection(dfni.index))
expect_ni_c = c*ni*total
print("stars in ni and carbon:", ni_c,"expected _ni_c value:", expect_ni_c )
ni_c_correlation = ((ni_c/expect_ni_c)*100)  -100
print('percent change abunance from random distribution:', ni_c_correlation, '%')

n_c = len(dfc.index.intersection(dfn.index))
expect_n_c = c*n*total
print("stars in Nitrogen and carbon:", n_c,"expected _n_c value:", expect_n_c )
n_c_correlation = ((n_c/expect_n_c)*100)  -100
print('percent change abunance from random distribution:', n_c_correlation, '%') 

ca_c = len(dfc.index.intersection(dfca.index))
expect_ca_c = c*ca*total
print("stars in calcium and carbon:", ca_c,"expected _ca_c value:", expect_ca_c )
ca_c_correlation = ((ca_c/expect_ca_c)*100)  -100
print('percent change abunance from random distribution:', ca_c_correlation, '%')

mg_c = len(dfc.index.intersection(dfmg.index))
expect_mg_c = c*mg*total
print("stars in mg and carbon:", mg_c,"expected _mg_c value:", expect_mg_c )
mg_c_correlation = ((mg_c/expect_mg_c)*100)  -100
print('percent change abunance from random distribution:', mg_c_correlation, '%')

n_ca = len(dfca.index.intersection(dfn.index))
expect_n_ca = ca*n*total
print("stars in Nitrogen and calcium:", n_ca,"expected _n_ca value:", expect_n_ca ) 
n_ca_correlation = ((n_ca/expect_n_ca)*100) -100
print('percent change abunance from random distribution:', n_ca_correlation, '%') 

n_mg = len(dfmg.index.intersection(dfn.index))
expect_n_mg = mg*n*total
print("stars in Nitrogen and mg:", n_mg,"expected _n_mg value:", expect_n_mg ) 
n_mg_correlation = ((n_mg/expect_n_mg)*100) -100
print('percent change abunance from random distribution:', n_mg_correlation, '%')

ca_mg = len(dfmg.index.intersection(dfca.index))
expect_ca_mg = mg*ca*total
print("stars in calcium and mg:", ca_mg,"expected _ca_mg value:", expect_ca_mg ) 
ca_mg_correlation = ((ca_mg/expect_ca_mg)*100) -100
print('percent change abunance from random distribution:', ca_mg_correlation, '%')









# In[14]:


# Define the categories and their corresponding values
categories = ['Ni_C', 'Ni_N', 'Ni_Ca', 'Ni_Mg', 'C_N', 'C_Ca', 'C_Mg', 'N_Ca', 'N_Mg', 'Ca_Mg']
percent_changes = [902.8733883407469, 23.139682033161037, 452.7554209429584, 902.8733883407469,
                   735.259871837106, 11878.288563960661, 491.350526727152, -83.4812232277509,
                   654.3098326254301, -52.02039914351204]

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the percent changes
ax.barh(categories, percent_changes, color='blue', label='Percent Change')

# Add labels and title
ax.set_xlabel('Percent Change from Expected (%)')
ax.set_ylabel('Category')
ax.set_title('Percent Change from Expected vs Category')
ax.legend()

# Show the plot
plt.show()

