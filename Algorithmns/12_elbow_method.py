from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = [15, 19, 15, 5, 13, 17, 15, 12, 8, 6, 9, 13]
x2 = [13, 16, 17, 6, 17, 14, 15, 13, 7, 6, 10, 12]

X = np.array(list(zip(x1, x2)))

distortions = []
K = range(1,8)
for i in K:
    print (i)
    model = KMeans(n_clusters=i)
    model.fit(X)
    distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.plot()
plt.plot(K, distortions, 'bx-')
plt.show()
