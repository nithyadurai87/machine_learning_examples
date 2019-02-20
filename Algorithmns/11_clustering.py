"""
import matplotlib.pyplot as plt

input_x1 = [7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3]
input_x2 = [5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7]

plt.figure()
plt.plot(input_x1,input_x2,'.')
plt.grid(True)
#plt.show()

K1 = 4
K2 = 5

c1_dis = []
c2_dis = []
for i,j in zip(input_x1,input_x2):
	c1_dis.append(i-K1)
	c2_dis.append(i-K2)
		
print (c1_dis,c2_dis)
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T
X = np.vstack((X, y)).T
K = range(1, 10)
meandistortions = []
for k in K:
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(X)
	meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
