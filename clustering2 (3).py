#!/usr/bin/env python
# coding: utf-8

# In[18]:


#!pip install scikit-learn

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[9]:


file_path = r"C:\Users\21rgo\Downloads\dr9_v2.0_MRS_stellar.csv.gz"

# Desired sample size as a percentage of the total number of rows
desired_sample_size_percentage = 1  # For example, 1%

# First, determine the total number of rows in the file
total_rows = sum(1 for row in open(file_path, 'r', encoding='latin-1')) - 1  # Exclude header row

# Calculate the number of rows to sample based on the desired percentage
num_rows_to_sample = int(total_rows * desired_sample_size_percentage / 100)

# Initialize an empty dataframe to hold the sampled rows
sampled_df = pd.DataFrame()

# The chunk size can be adjusted based on your system's memory capacity
chunk_size = 10000

# Calculate sampling fraction for each chunk
# Ensuring at least 1 row per chunk is sampled for very small fractions
sample_fraction = max(1, num_rows_to_sample) / total_rows

sampled_rows_count = 0
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Sample from each chunk
    chunk_sample = chunk.sample(frac=sample_fraction, replace=False)
    sampled_rows_count += len(chunk_sample)
   
    # Concatenate the sampled rows
    sampled_df = pd.concat([sampled_df, chunk_sample])
   
    # If we've collected more or equal rows than desired, break the loop
    if sampled_rows_count >= num_rows_to_sample:
        break

# If we have sampled more rows than desired, randomly drop the extra rows
if len(sampled_df) > num_rows_to_sample:
    sampled_df = sampled_df.sample(n=num_rows_to_sample)

# Save the subsample to a new CSV file
sampled_df.to_csv("C:/Users/21rgo/Downloads/star_sample.csv", index=False)


# In[10]:


#bigpath = r"C:\Users\21rgo\Downloads\dr9_v2.0_MRS_catalogue.csv.gz"
bigpath = r"C:\Users\21rgo\Downloads\dr9_v2.0_MRS_stellar.csv.gz"

path = "C:/Users/21rgo/Downloads/star_sample.csv"
df = pd.read_csv(bigpath)


# In[11]:


print(df.columns)


# In[26]:


# Convert 'feh_cnn' and 'alpha_m_lasp' to numeric values
df['feh_cnn'] = pd.to_numeric(df['feh_cnn'], errors='coerce')
df['alpha_m_lasp'] = pd.to_numeric(df['alpha_m_lasp'], errors='coerce')

df['feh_cnn'] = df['feh_cnn'].replace(-9999.00, pd.NA)
df['alpha_m_lasp'] = df['alpha_m_lasp'].replace(-9999.00, pd.NA)

# Remove rows with missing values
df = df.dropna(subset=['feh_cnn', 'alpha_m_lasp'])

# Extracting the data
x = df['feh_cnn'].values
y = df['alpha_m_lasp'].values

# Compute Gaussian KDE
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

# Creating the figure
plt.figure(figsize=(10, 6))

# Plot the heatmap using contourf for the density
contour = plt.contourf(xx, yy, f, cmap='viridis')

# Add colorbar with label
plt.colorbar(contour, label='Probability Density')

# Optionally overlay the scatter plot with low alpha for reference
#plt.scatter(x, y, s=1, color='black', alpha=0.05)

# Adding labels and title
plt.xlabel('FeH CNN')
plt.ylabel('Alpha_m_lasp')
plt.title('Density Heatmap of Alpha_m_lasp vs FeH CNN (Gaussian KDE)')

# Displaying the plot
plt.grid(True)
plt.show()

num_points = len(x)
print("Number of points plotted:", num_points)


# In[27]:


# Select the columns for clustering
X = np.column_stack((feh_cnn, alpha_m_lasp))#this was my first clustering demo
#X = X[:len(df)]
df = df[:len(X)]
# Choose the number of clusters (adjustable)
num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, init='random', random_state=42)  #k-means++
df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(feh_cnn,alpha_m_lasp, c= df['cluster'],  cmap='viridis',s=1)
plt.title('K-means Clustering')
plt.xlabel('feh_cnn')
plt.ylabel('alpha_m_lasp')
plt.show()

num_points = len(feh_cnn)
print("Number of points plotted:", num_points)


# In[28]:


#account for missing points
df['gaia_g_mean_mag'] = df['gaia_g_mean_mag'].replace(-9999.00, pd.NA)
df['logg_lasp'] = df['logg_lasp'].replace(-9999.00, pd.NA)
df['rv_br1'] = df['rv_br1'].replace(-9999.00, pd.NA)

df['gaia_g_mean_mag'].replace('NAType', pd.NA, inplace=True)
df['logg_lasp'].replace('NAType', pd.NA, inplace=True)
df['rv_br1'].replace('NAType', pd.NA, inplace=True)

gaia_g_mean_mag = df.dropna(subset=['gaia_g_mean_mag'])
logg_lasp = df.dropna(subset=['logg_lasp'])
rv_br1 = df.dropna(subset=['rv_br1'])

gaia_g_mean_mag = pd.to_numeric(df['gaia_g_mean_mag'], errors='coerce')
gaia_g_mean_mag = gaia_g_mean_mag[gaia_g_mean_mag.notna()].astype(float)

logg_lasp = pd.to_numeric(df['logg_lasp'], errors='coerce')
logg_lasp = logg_lasp[logg_lasp.notna()].astype(float)

rv_br1 = pd.to_numeric(df['rv_br1'], errors='coerce')
rv_br1 = rv_br1[rv_br1.notna()].astype(float)

length_gaia_g_mean_mag = len(gaia_g_mean_mag)
length_logg_lasp = len(logg_lasp)
length_rv_br1 = len(rv_br1)

min_length = min(length_gaia_g_mean_mag, length_logg_lasp, length_rv_br1 )


gaia_g_mean_mag = gaia_g_mean_mag[:min_length]
logg_lasp = logg_lasp[:min_length]
rv_br1 = rv_br1[:min_length]

data = {
    'gaia_g_mean_mag': gaia_g_mean_mag[:min_length],
    'logg_lasp': logg_lasp[:min_length],
    'rv_br1': rv_br1[:min_length]}


# Create a new DataFrame from the dictionary
g_test_df = pd.DataFrame(data)
print(len(g_test_df))
print(min_length)




# In[29]:


# Select the columns for clustering
X = np.column_stack((gaia_g_mean_mag, logg_lasp, rv_br1))  # Adding the third dimension 'dfn'
g_test_df = g_test_df[:len(X)]

# Choose the number of clusters (adjustable)
num_clusters = 4

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
g_test_df['cluster'] = kmeans.fit_predict(X)

# Create figure and subplots
fig = plt.figure(figsize=(12, 5))

# First subplot: default perspective
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(gaia_g_mean_mag, logg_lasp, rv_br1, c=g_test_df['cluster'], cmap='viridis', s=1)
ax1.set_xlabel('gaia_g_mean_mag')
ax1.set_ylabel('logg_lasp')
ax1.set_zlabel('rv_br1')
ax1.set_title('Default Perspective')

# Second subplot: inverse Logg_lasp
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(gaia_g_mean_mag, logg_lasp, rv_br1, c=g_test_df['cluster'], cmap='viridis', s=1)
ax2.view_init(elev=30, azim=135)  # Change the elevation and azimuth angles for a different perspective
ax2.set_xlabel('gaia_g_mean_mag')
ax2.set_ylabel('logg_lasp')
ax2.set_zlabel('rv_br1')
ax2.set_title('Different Perspective')

plt.tight_layout()
plt.show()


# In[30]:


# Select the columns for clustering
X = np.column_stack((gaia_g_mean_mag, logg_lasp, rv_br1))

# Rescale rv_br1 and logg_lasp to match the range of gaia_g_mean_mag
scaler = MinMaxScaler(feature_range=(0, 14))
X_scaled = scaler.fit_transform(X)
gaia_g_mean_mag_scaled = gaia_g_mean_mag  # Assuming it's already in the desired range [0, 14]
logg_lasp_scaled = X_scaled[:, 1]
rv_br1_scaled = X_scaled[:, 2]

num_clusters = 4

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
g_test_df['cluster'] = kmeans.fit_predict(X_scaled)

fig = plt.figure(figsize=(12, 5))

# First subplot: default perspective
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(gaia_g_mean_mag_scaled, logg_lasp_scaled, rv_br1_scaled, c=g_test_df['cluster'], cmap='viridis', s=1)
ax1.set_xlabel('gaia_g_mean_mag')
ax1.set_ylabel('logg_lasp')
ax1.set_zlabel('rv_br1')
ax1.set_title('Default Perspective')

# Second subplot: different perspective
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(gaia_g_mean_mag_scaled, logg_lasp_scaled, rv_br1_scaled, c=g_test_df['cluster'], cmap='viridis', s=1)
ax2.view_init(elev=30, azim=135)  # Change the elevation and azimuth angles for a different perspective
ax2.set_xlabel('gaia_g_mean_mag')
ax2.set_ylabel('logg_lasp')
ax2.set_zlabel('rv_br1')
ax2.set_title('Different Perspective')

plt.tight_layout()
plt.show()


# In[31]:


# account for the null values

df['feh_cnn'].replace(-9999.00, pd.NA, inplace=True)
df['feh_cnn'].replace('NAType', pd.NA, inplace=True)
dffeh = df.dropna(subset=['feh_cnn'])


df['ra_obs'].replace(-9999.00, pd.NA, inplace=True)
df['ra_obs'].replace('NAType', pd.NA, inplace=True)
ra_obs = df.dropna(subset=['ra_obs'])

ra_obs = pd.to_numeric(df['ra_obs'], errors='coerce')
ra_obs1 = ra_obs[ra_obs.notna()].astype(float)

df['dec_obs'].replace(-9999.00, pd.NA, inplace=True)
df['dec_obs'].replace('NAType', pd.NA, inplace=True)
dec_obs = df.dropna(subset=['dec_obs'])

dec_obs = pd.to_numeric(df['dec_obs'], errors='coerce')
dec_obs1 = dec_obs[dec_obs.notna()].astype(float)


df['feh_cnn'] = df['feh_cnn'].replace(-9999.00, pd.NA)
df['c_fe'] = df['c_fe'].replace(-9999.00, pd.NA)
df['n_fe'] = df['n_fe'].replace(-9999.00, pd.NA)
df['mg_fe'] = df['mg_fe'].replace(-9999.00, pd.NA)
df['al_fe'] = df['al_fe'].replace(-9999.00, pd.NA)
df['ca_fe'] = df['ca_fe'].replace(-9999.00, pd.NA)
df['ni_fe'] = df['ni_fe'].replace(-9999.00, pd.NA)
df['alpha_m_lasp'] = df['alpha_m_lasp'].replace(-9999.00, pd.NA)



df['c_fe'].replace('NAType', pd.NA, inplace=True)
df['n_fe'].replace('NAType', pd.NA, inplace=True)
df['mg_fe'].replace('NAType', pd.NA, inplace=True)
df['al_fe'].replace('NAType', pd.NA, inplace=True)
df['ca_fe'].replace('NAType', pd.NA, inplace=True)
df['ni_fe'].replace('NAType', pd.NA, inplace=True)
df['alpha_m_lasp'].replace('NAType', pd.NA, inplace=True)




# Drop rows with NaN values in the 'rv_r0' and 'rv_b0' columns
dfc = df.dropna(subset=['c_fe'])
dfn = df.dropna(subset=['n_fe'])
dfmg = df.dropna(subset=['mg_fe'])
dfal = df.dropna(subset=['al_fe'])
dfca = df.dropna(subset=['ca_fe'])
dfni = df.dropna(subset=['ni_fe'])
alpha_m = df.dropna(subset=['alpha_m_lasp'])



# In[33]:


#non numeric inputs are set aside

dfc = pd.to_numeric(df['c_fe'], errors='coerce')
dfc = dfc[dfc.notna()].astype(float)

dfn = pd.to_numeric(df['n_fe'], errors='coerce')
dfn = dfn[dfn.notna()].astype(float)

dfmg = pd.to_numeric(df['mg_fe'], errors='coerce')
dfmg = dfmg[dfmg.notna()].astype(float)

dffeh = pd.to_numeric(df['feh_cnn'], errors='coerce')
dffeh = dffeh[dffeh.notna()].astype(float)

dfni = pd.to_numeric(df['ni_fe'], errors='coerce')
dfni = dfni[dfni.notna()].astype(float)

dfca = pd.to_numeric(df['ca_fe'], errors='coerce')
dfca = dfca[dfca.notna()].astype(float)

dfal = pd.to_numeric(df['al_fe'], errors='coerce')
dfal = dfal[dfal.notna()].astype(float)

df_alpha_m_lasp = pd.to_numeric(df['alpha_m_lasp'], errors='coerce')
df_alpha_m_lasp = dfal[dfal.notna()].astype(float)


# Get the lengths of all arrays
length_dffeh = len(dffeh)
length_dfc = len(dfc)
length_dfn = len(dfn)
length_dfmg = len(dfmg)
length_dfni = len(dfni)
length_dfca = len(dfca)
length_dfal = len(dfal)
length_df_alpha_m_lasp = len(df_alpha_m_lasp)
length_dec_obs = len(dec_obs)
length_ra_obs = len(ra_obs)


# Find the minimum length
# Calculate the minimum length among all variables
min_length = min(length_dffeh, length_dfc, length_dfn, length_dfmg, length_dfni, length_dfca, length_dfal, length_dec_obs, length_ra_obs)

# Determine which variable corresponds to the minimum length
min_length_variable = {
    length_dffeh: 'dffeh',
    length_dfc: 'dfc',
    length_dfn: 'dfn',
    length_dfmg: 'dfmg',
    length_dfni: 'dfni',
    length_dfca: 'dfca',
    length_dfal: 'dfal',
    length_dec_obs: 'dec_obs',
    length_ra_obs: 'ra_obs'
}[min_length]

# Print the minimum length and the corresponding variable
print(f"Minimum length: {min_length}, Variable: {min_length_variable}")

# Truncate all arrays to match the size of the smallest array
dffeh = dffeh[:min_length]
dfc = dfc[:min_length]
dfn = dfn[:min_length]
dfmg = dfmg[:min_length]
dfni = dfni[:min_length]
dfca = dfca[:min_length]
dfal = dfal[:min_length]
df_alpha_m_lasp = df_alpha_m_lasp[:min_length]
dec_obs = dec_obs[:min_length]
ra_obs = ra_obs[:min_length]

min_length


# In[34]:


# Calculate the minimum length among all variables
min_length = min(length_dffeh, length_dfc, length_dfn, length_dfmg, length_dfni, length_dfca, length_dfal, length_dec_obs, length_ra_obs)

# Determine which variable corresponds to the minimum length
min_length_variable = {
    length_dffeh: 'dffeh',
    length_dfc: 'dfc',
    length_dfn: 'dfn',
    length_dfmg: 'dfmg',
    length_dfni: 'dfni',
    length_dfca: 'dfca',
    length_dfal: 'dfal',
    length_dec_obs: 'dec_obs',
    length_ra_obs: 'ra_obs'
}[min_length]

# Print the minimum length and the corresponding variable
print(f"Minimum length: {min_length}, Variable: {min_length_variable}")


# In[ ]:





# In[35]:


data = {
    'dffeh': dffeh[:min_length],
    'dfc': dfc[:min_length],
    'dfn': dfn[:min_length],
    'dfmg': dfmg[:min_length],
    'dfni': dfni[:min_length],
    'dfca': dfca[:min_length],
    'dfal': dfal[:min_length],
    
}

# Create a new DataFrame from the dictionary
graph_df = pd.DataFrame(data)
graph_df = graph_df[:min_length]
len(graph_df)

df = df[:338813]

#146734
df


# In[36]:


x = ra_obs
y = dec_obs

# Create a scatter plot with a heatmap
plt.hexbin(x, y, gridsize=80, cmap='inferno')
plt.colorbar(label='count in bin')  # Add a colorbar
plt.xlabel('Right Ascension')
plt.ylabel('Declination')
plt.title('Positional Heatmap of data Locations')
plt.show()

counts, xedges, yedges = np.histogram2d(x, y, bins=80)

# Find the indices of the bins with the highest counts
top_indices = np.unravel_index(np.argsort(counts.ravel())[-10:], counts.shape)

# Get the coordinates and counts of the bins
top_coordinates = [(xedges[top_indices[0][i]], yedges[top_indices[1][i]]) for i in range(10)]
top_counts = [counts[top_indices[0][i], top_indices[1][i]] for i in range(10)]

# Print the coordinates and counts of the six most concentrated bins
print("Coordinates and counts of the six most concentrated bins:")
for i, (coord, count) in enumerate(zip(top_coordinates, top_counts), 1):
    print(f"Bin {i}: x = {coord[0]}, y = {coord[1]}, count = {count}")



# In[43]:


# Select the columns for clustering
X = np.column_stack((dffeh, dfc))

# Scale N/Fe to have a range of 3
scaler = MinMaxScaler(feature_range=(0, 3))
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1, 1)).reshape(-1)

# Choose the number of clusters (you can adjust this)
num_clusters = 8

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
graph_df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster_label in range(num_clusters):
    cluster_indices = graph_df[graph_df['cluster'] == cluster_label].index
    plt.scatter(dffeh[cluster_indices], dfc[cluster_indices], label=f'Cluster {cluster_label}', s=10)

plt.title('K-means Clustering: Metallicity by Carbon Content')
plt.xlabel('Fe/H')
plt.ylabel('C/Fe (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# Print number of plotted points
num_points = len(X)
print("Number of points plotted:", num_points)


# In[ ]:





# In[21]:


#what fraction of the stars fall into each of the selected cluster? %
#we find one value for expected amount of N in C and one for C in N
#what percent over or under these expected value is present?


# In[38]:


# Select the columns for clustering
X = np.column_stack((dffeh, dfn))

# Scale N/Fe to have a range of 3
scaler = MinMaxScaler(feature_range=(0, 3))
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1, 1)).reshape(-1)

# Choose the number of clusters (you can adjust this)
num_clusters = 8

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
graph_df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster_label in range(num_clusters):
    cluster_indices = graph_df[graph_df['cluster'] == cluster_label].index
    plt.scatter(dffeh[cluster_indices], dfn[cluster_indices], label=f'Cluster {cluster_label}', s=1)

plt.title('K-means Clustering')
plt.xlabel('Fe/H')
plt.ylabel('N/Fe (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# Print number of plotted points
num_points = len(X)
print("Number of points plotted:", num_points)


# In[1]:


# Select the columns for plotting
X = np.column_stack((dffeh, dfn))

# Scale N/Fe to have a range of 3
scaler = MinMaxScaler(feature_range=(0, 3))
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1, 1)).reshape(-1)

# Create a scatter plot with density heatmap
plt.figure(figsize=(10, 8))
sns.kdeplot(x=dffeh, y=dfn, cmap='viridis', shade=True, bw_adjust=0.5)

plt.title('Scatter Plot with Heatmap (Density Plot)')
plt.xlabel('Fe/H')
plt.ylabel('N/Fe (Scaled)')
plt.grid(True)

plt.show()

# Print number of plotted points
num_points = len(X)
print("Number of points plotted:", num_points)


# In[41]:


# Select the columns for clustering
X = np.column_stack((dffeh, dfni))
#X = X[:len(df)]
df = df[:len(X)]
# Choose the number of clusters (you can adjust this)
num_clusters = 6

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(dffeh,dfni, c= df['cluster'], cmap='viridis',s=1)
plt.title('K-means Clustering')
plt.xlabel('Fe/H')
plt.ylabel('Ni/Fe')
plt.show()

#print(df['gaia_g_mean_mag'])



# In[42]:


# Select the columns for clustering
X = np.column_stack((dffeh, dfca))
#X = X[:len(df)]
df = df[:len(X)]
# Choose the number of clusters (you can adjust this)
num_clusters = 6

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(dffeh,dfca, c= df['cluster'], cmap='viridis',s=1)
plt.title('K-means Clustering')
plt.xlabel('Fe/H')
plt.ylabel('Ca/Fe')
plt.show()


# In[43]:


# Select the columns for clustering
X = np.column_stack((dffeh, dfal))
#X = X[:len(df)]
df = df[:len(X)]
# Choose the number of clusters (you can adjust this)
num_clusters = 12

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(dffeh,dfal, c= df['cluster'], cmap='viridis',s=1)
plt.title('K-means Clustering')
plt.xlabel('Fe/H')
plt.ylabel('Al/Fe')
plt.show()

#print(df['gaia_g_mean_mag'])



# In[44]:


# Select the columns for clustering
X = np.column_stack((dffeh, dfmg))

# Scale N/Fe to have a range of 3
scaler = MinMaxScaler(feature_range=(0, 3))
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1, 1)).reshape(-1)

# Choose the number of clusters (you can adjust this)
num_clusters = 8

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
graph_df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster_label in range(num_clusters):
    cluster_indices = graph_df[graph_df['cluster'] == cluster_label].index
    plt.scatter(dffeh[cluster_indices], dfmg[cluster_indices], label=f'Cluster {cluster_label}', s=1)

plt.title('K-means Clustering')
plt.xlabel('Fe/H')
plt.ylabel('Mg/Fe (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# Print number of plotted points
num_points = len(X)
print("Number of points plotted:", num_points)


# In[46]:


from mpl_toolkits.mplot3d import Axes3D  # Importing this for 3D plotting

# Select the columns for clustering
X = np.column_stack((dffeh, dfmg, dfn))  # Adding the third dimension 'dfn'

# Choose the number of clusters (you can adjust this)
num_clusters = 6

# Apply KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Create figure and subplots
fig = plt.figure(figsize=(12, 5))

# First subplot: default perspective
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(dffeh, dfmg, dfn, c=df['cluster'], cmap='viridis', s=1)
ax1.set_xlabel('Fe/H')
ax1.set_ylabel('Mg/Fe')
ax1.set_zlabel('N/Fe')
ax1.set_title('Default Perspective')

# Second subplot: different perspective
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(dffeh, dfmg, dfn, c=df['cluster'], cmap='viridis', s=1)
ax2.view_init(elev=30, azim=135)  # Change the elevation and azimuth angles for a different perspective
ax2.set_xlabel('Fe/H')
ax2.set_ylabel('Mg/Fe')
ax2.set_zlabel('N/Fe')
ax2.set_title('Different Perspective')

plt.tight_layout()
plt.show()

