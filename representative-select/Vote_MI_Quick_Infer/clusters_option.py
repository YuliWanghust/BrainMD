import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# Load the provided CSV file
file_path = r'E:\Yuli\Projects\FM\FM_selection\VAE-mnist-tf-main\results_100epoches_64batch\PMLR_map_epoch_70.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns for clustering
X = data[['z1', 'z2']]

# Perform K-means clustering with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(X)

# Plot the K-means clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['z1'], data['z2'], c=data['kmeans_cluster'], cmap='viridis', marker='o', alpha=0.6, edgecolor='w', s=10)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')  # cluster centers
plt.title('K-means Clustering with 10 Clusters')
plt.xlabel('z1')
plt.ylabel('z2')
plt.colorbar(label='Cluster Label')
plt.show()

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=50)
data['dbscan_cluster'] = dbscan.fit_predict(X)

# Print the number of unique labels (clusters)
unique_labels = len(set(data['dbscan_cluster']))
print(f'Number of unique labels (clusters): {unique_labels}')

# Save the updated DataFrame back to CSV
output_csv_path = r'E:\Yuli\Projects\FM\FM_selection\VAE-mnist-tf-main\results\PMLR_map_epoch_70.csv'  # Replace with the desired output CSV file path
data.to_csv(output_csv_path, index=False)

# Plot the DBSCAN clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['z1'], data['z2'], c=data['dbscan_cluster'], cmap='plasma', marker='o', alpha=0.6, edgecolor='w', s=10)
plt.title('DBSCAN Clustering')
plt.xlabel('z1')
plt.ylabel('z2')
plt.colorbar(label='Cluster Label')
plt.show()
