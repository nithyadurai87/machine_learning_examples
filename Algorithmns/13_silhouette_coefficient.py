import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
plt.subplot(3, 2, 1)

x1 = [15, 19, 15, 5, 13, 17, 15, 12, 8, 6, 9, 13]
x2 = [13, 16, 17, 6, 17, 14, 15, 13, 7, 6, 4, 12]
plt.title('Instances')
plt.scatter(x1, x2)

X = np.array(list(zip(x1, x2)))

c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
m = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

p = 1
for t in [2, 3, 4, 5, 8]:
	p += 1
	plt.subplot(3, 2, p)
	kmeans_model = KMeans(n_clusters=t).fit(X)
	for i, j in enumerate(kmeans_model.labels_):
		plt.plot(x1[i], x2[i], color=c[j], marker=m[j],ls='None')
	plt.title('K = %s, SC = %.03f' % (t, metrics.silhouette_score(X, kmeans_model.labels_,metric='euclidean')))
plt.show()