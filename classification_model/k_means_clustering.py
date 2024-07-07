import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = []
        self.data_mean = None
        
    def generate_random_centroids(self, latitude_data, longitude_data):
        x_co = np.random.uniform(np.amin(latitude_data), np.amax(latitude_data), size=self.k)
        y_co = np.random.uniform(np.amin(longitude_data), np.amax(longitude_data), size=self.k)
        self.centroids = list(zip(x_co, y_co))        

    def finding_nearest_centroid_index(self, old_centroids, data_points):
        nearest_centroid_index = []
        for data_point in data_points:
            distances = np.sqrt(np.sum((old_centroids - data_point) ** 2, axis=1))
            nearest_centroid_index.append(np.argmin(distances))
        return nearest_centroid_index

    def forming_clusters(self, data, labels):
        clusters = []
        for i in range(self.k):
            clusters.append(np.argwhere(np.array(labels) == i).flatten())
        return clusters
        
    def regenerating_centroids(self, clusters, data):
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                points = data[cluster]
                mean_x = np.mean(points[:, 0])
                mean_y = np.mean(points[:, 1])
                new_centroids.append([mean_x, mean_y])
            else:
                
                new_centroids.append(self.data_mean)
        return new_centroids

    def fit(self, data):
        latitude_data = data.apply(lambda x: x[0])
        longitude_data = data.apply(lambda x: x[1])
        self.data_mean = [np.mean(latitude_data), np.mean(longitude_data)]
        self.generate_random_centroids(latitude_data, longitude_data)
        lat_and_long = np.array(list(zip(latitude_data, longitude_data)))

        while True:
            old_centroids = np.array(self.centroids)
            nearest_centroid_indices = self.finding_nearest_centroid_index(np.array(self.centroids), lat_and_long)
            clusters = self.forming_clusters(lat_and_long, nearest_centroid_indices)
            self.centroids = self.regenerating_centroids(clusters, lat_and_long)
            centroid_shift = np.sum((np.array(self.centroids) - old_centroids) ** 2)
            if centroid_shift < 0.00001:
                break
        return self.centroids, nearest_centroid_indices

    def calculate_sum_of_squared_distances(self, data):
        lat_and_long = np.array(list(data))
        nearest_centroids = self.finding_nearest_centroid_index(np.array(self.centroids), lat_and_long)
        sum_of_squared_distances = 0
        for i, centroid in enumerate(self.centroids):
            cluster_points = lat_and_long[np.array(nearest_centroids) == i]
            distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
            sum_of_squared_distances += np.sum(distances ** 2)
        return sum_of_squared_distances

def scaling_data(scaling_constant, data):
    return (data - np.amin(data)) / scaling_constant

def calculate_elbow(k_values, data):
    WCSS = []
    for k in k_values:
        kmeans = KMeansClustering(k)
        kmeans.fit(data)
        WCSS.append(kmeans.calculate_sum_of_squared_distances(data))
    print(WCSS)
    return WCSS

# Load data from a csv file which contains only home state details
HomeState_data = pd.read_csv("HomeState_data.csv")

# Scaling latitude and longitude data

minimum_latitude = np.amin(HomeState_data['Latitude'])
maximum_latitude = np.amax(HomeState_data['Latitude'])
minimum_longitude = np.amin(HomeState_data['Longitude'])
maximum_longitude = np.amax(HomeState_data['Longitude'])

scaling_constant = max((maximum_longitude - minimum_longitude), (maximum_latitude - minimum_latitude))
#Scaling constant used for both latitude and longitude data is same to maintain the distance ratios as it is between the points 
scaled_latitude_data = scaling_data(scaling_constant, HomeState_data['Latitude'])
scaled_longitude_data = scaling_data(scaling_constant, HomeState_data['Longitude'])

scaled_lat_and_long = pd.Series(list(zip(scaled_latitude_data, scaled_longitude_data)))

# Calculate the elbow method to choose k value
k_values = range(1, 20)
sum_of_squared_distances = calculate_elbow(k_values, scaled_lat_and_long)

# Plot the curve for elbow method
plt.plot(k_values, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

print("From Elbow method we can choose k=6 as the optimal K value")


k = 6
kmeans = KMeansClustering(k)
centroids, labels = kmeans.fit(scaled_lat_and_long)
print("Centroids for scaled latitudes and longitudes:\n", kmeans.centroids)

# Plotting the clusters
centroids_x_co = [centroid[0] for centroid in centroids]
centroids_y_co = [centroid[1] for centroid in centroids]
latitude_data = scaled_lat_and_long.apply(lambda x: x[0])
longitude_data = scaled_lat_and_long.apply(lambda x: x[1])

plt.scatter(latitude_data, longitude_data, c=labels)
plt.scatter(centroids_x_co, centroids_y_co, c='black', marker='*', s=200)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('K-means Clustering')
plt.show()

