#!/usr/bin/env python
# coding: utf-8

# In[9]:


#This log prints out candadates in our data base and reveals duplicates at the bottom to skip o3ver because they have 
#already been assesed
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import numpy as np
import gspread
from tabulate import tabulate

df = pd.read_csv("C:\\Users\\21rgo\\The Plus 1\\astro research\\Dwarf Search Scanner - Sheet1.csv" ,sep=',')


# In[10]:


# Make sure you have the proper permissions to access the Google Sheets document.
url = 'https://docs.google.com/spreadsheets/d/1kWgeGHc0LbwUCcelLJImgYr177pP1PQOP6uPStLMq1A/edit#gid=0'

# Authenticate and open the Google Sheets document using gspread
gc = gspread.service_account(filename='C:\\Users\\21rgo\\Downloads\\single-sheet-scanner-c916840af6c0.json')
doc = gc.open_by_url(url)


# Read all sheets from the Google Sheets document into a dictionary of DataFrames
dfs_dict = {}
for sheet in doc.worksheets():
    sheet_name = sheet.title
    data = sheet.get_all_values()
    df = pd.DataFrame(data[2:], columns=data[1])  # Assuming the first row contains column headers
    dfs_dict['Rowan'] = df

# Access a specific sheet's DataFrame by its name, e.g., 'Sheet1'
sheet_name = 'Sheet1'  # Replace 'Sheet1' with the actual sheet name you want to access.
df = dfs_dict['Rowan']

# Now you have the data from the specified sheet in the DataFrame 'df'
#print(df.head())

print(dfs_dict.keys())


# In[6]:


#scroll to bottom to see if your newly imported rows have already been charted

# Define the number of decimal places for rounding
decimal_places = 2
df.index -= 3 # only change if rows are misaligned

df['RA'] = pd.to_numeric(df['RA'], errors='coerce')
df['Dec'] = pd.to_numeric(df['Dec'], errors='coerce')

# Round the 'RA' and 'Dec' columns to the specified decimal places
df['RA'] = df['RA'].apply(lambda x: round(float(x), decimal_places))
df['Dec'] = df['Dec'].apply(lambda x: round(float(x), decimal_places))

# Identify duplicate rows based on 'RA' and 'Dec' columns, including NaN values
duplicate_rows = df[df.duplicated(subset=["RA", "Dec"], keep=False) | df[["RA", "Dec"]].isna().any(axis=1)]

# Drop rows with NaN values from duplicate rows
duplicate_rows_without_nan = duplicate_rows.dropna(subset=["RA", "Dec"])

# Display duplicate rows using tabulate
print("Duplicate rows based on limited RA and Dec, ignoring NaN values:")
print(tabulate(duplicate_rows_without_nan, headers="keys", tablefmt="psql"))

# row five should be the first dup


# In[13]:


#command find your duplicated rows to see where/who else found your same candidate.

# Convert 'RA' and 'Dec' columns to numeric values
df['RA'] = pd.to_numeric(df['RA'], errors='coerce')
df['Dec'] = pd.to_numeric(df['Dec'], errors='coerce')
import pandas as pd
get_ipython().system('pip install tabulate')
from tabulate import tabulate
decimal_places = 2

df.index -= 3 #Only alter this number if the rows become misaligned

#Lrounds decimals
df['RA'] = df['RA'].apply(lambda x:round(float(x), decimal_places))
df['Dec'] = df['Dec'].apply(lambda x:round(float(x), decimal_places))

duplicate_rows = df[df.duplicated(subset=["RA", "Dec"], keep=False) | df[["RA", "Dec"]].isna().any(axis=1)]
duplicate_rows_without_nan = duplicate_rows.dropna(subset=["RA", "Dec"])


print("Duplicates based on limited RA and Dec, ignoring NaN values:")
duplicate_rows_without_nan


df["Limited RA"] = df["RA"].apply(lambda x: round(float(x), decimal_places))
df["Limited Dec"] = df["Dec"].apply(lambda x: round(float(x), decimal_places))


# Identify duplicates based on the limited RA and Dec values
duplicate_groups = df[df.duplicated(subset=["Limited RA", "Limited Dec"], keep=False)].groupby(["Limited RA", "Limited Dec"])

# Display duplicates in pairs in the opposite order using tabulate
duplicate_pairs = list(duplicate_groups)
for (limited_ra, limited_dec), group in duplicate_groups:
    print(f"Duplicate pair (RA: {limited_ra}, Dec: {limited_dec}):")
    print(group)
    print("\n")

#first pair should be row 22 and 194

