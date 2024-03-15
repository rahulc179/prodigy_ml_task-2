#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the mall customer data
mall_data = pd.read_csv('mall_customers.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(mall_data.head())

# Extracting features (Annual Income and Spending Score)
X = mall_data.iloc[:, [3, 4]].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve to find the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow curve, let's choose 5 clusters

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Add cluster labels to the dataset
mall_data['Cluster'] = cluster_labels

# Display the clustered data
print("\nClustered Mall Customer Data:")
print(mall_data.head())
