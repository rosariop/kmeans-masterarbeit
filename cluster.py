import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the JSON data
with open('company_data.json', 'r') as file:
    data = json.load(file)

# Extract the relevant attributes for clustering
attributes = ['customer_visits', 'sales_volume', 'expected_sales_volume', 'margin', 'service_expense',
              'ordering_interval']

# Define weights for each attribute
weights = [0.3, 0.2, 0.15, 0.2, 0.1, 0.05]

# Create a weighted data matrix
company_data = []
for company in data:
    weighted_attributes = [weights[i] * company[attr] for i, attr in enumerate(attributes)]
    company_data.append(weighted_attributes)

# Standardize the data (important for K-Means)
scaler = StandardScaler()
company_data_scaled = scaler.fit_transform(company_data)

# Choose the number of clusters (K)
K = 5  # You can adjust this based on your specific requirements

# Create and fit the K-Means model
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(company_data_scaled)

# Add cluster labels to the original data
for i, company in enumerate(data):
    company['cluster'] = kmeans.labels_[i]

# Calculate cluster centroids
cluster_centers = kmeans.cluster_centers_

# Calculate the radius for the circles (e.g., standard deviation of each cluster's points)
cluster_radii = []
for cluster_num in range(K):
    cluster_points = company_data_scaled[kmeans.labels_ == cluster_num]
    cluster_radius = np.std(cluster_points, axis=0).max()
    cluster_radii.append(cluster_radius)

# Create scatter plots for visualization
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Define different colors for clusters

for cluster_num in range(K):
    cluster_data = [company for company in data if company['cluster'] == cluster_num]
    cluster_values = np.array(
        [company_data_scaled[i] for i, company in enumerate(data) if company['cluster'] == cluster_num])

    # Increase circle size for clusters 2 and 3
    circle_size = cluster_radii[cluster_num] * (2 if cluster_num in [2, 3] else 1)

    # Plot the data points
    plt.scatter(cluster_values[:, 0], cluster_values[:, 1], label=f'Cluster {cluster_num}', c=colors[cluster_num])

    # Plot circles around cluster centroids
    circle = plt.Circle((cluster_centers[cluster_num, 0], cluster_centers[cluster_num, 1]), circle_size,
                        color=colors[cluster_num], alpha=0.2)
    plt.gca().add_patch(circle)

    # Add company names as labels to data points
    for company in cluster_data:
        plt.annotate(company['id'], (cluster_values[cluster_data.index(company), 0], cluster_values[cluster_data.index(company), 1]))

plt.xlabel('Weighted Customer Visits (Standardized)')
plt.ylabel('Weighted Sales Volume (Standardized)')
plt.title('Weighted K-Means Clustering of Companies with Variable Circle Sizes and Labels')
plt.legend()
plt.savefig('sales_clusters.png')
plt.show()
