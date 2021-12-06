# import needed libraries
import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt

# setting a path to our dataset
path_to_dataset = './checkins.dat'
df = pd.read_csv(path_to_dataset,
                 sep="\|\s+",
                 skiprows=2,
                 names=['id', 'user_id', 'venue_id', 'latitude', 'longitude', 'created_at'],
                 engine='python')

# Remove rows with a NaN
df = df.dropna()
# and useless columns
df = df.drop(['id', 'user_id', 'venue_id', 'created_at'], axis=1)

# Truncate the dataset to 100000 values.
df = df.head(100000)

# do a clusterization by MeanShift method with a bandwidth=0.1
ms = MeanShift(bandwidth=0.1)
ms.fit(df.to_numpy())

centroids = ms.cluster_centers_
labels = ms.labels_

cluster_num = ms.predict(np.reshape([45.523452, -122.676207], (1, -1)))

x = centroids[:, :1]
y = centroids[:, 1:]

# Make a graphic visually similar to an Earth map.
plt.scatter(y, x)

cluster_num_dict = {}
for coord in df.values.tolist():
    cluster_num = ms.predict(np.reshape([coord], (1, -1)))[0]
    if cluster_num_dict.get(cluster_num):
        cluster_num_dict[cluster_num] += 1
    else:
        cluster_num_dict.update({cluster_num: 1})

max20 = []


def get_20_max():
    if len(max20) >= 20:
        return
    max_num = max(cluster_num_dict.values())
    max_cluster_num = list(cluster_num_dict.keys())[list(cluster_num_dict.values()).index(max_num)]
    max20.append(max_cluster_num)
    cluster_num_dict.pop(max_cluster_num)
    get_20_max()


get_20_max()

top20centroids = np.array([centroids[maximum] for maximum in max20])

x = top20centroids[:, :1]
y = top20centroids[:, 1:]
# Make a graphic with a cluster's centers.
plt.scatter(y, x)

# Code below is for comfortable coords input in https://www.mapcustomizer.com/

for x1, y1 in top20centroids:
    print(f'{x1},{y1}')

# Finally, we got a couples of coords which means a most useful places to make a billboards.
# (actually, for this dataset)
