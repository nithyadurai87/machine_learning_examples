import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
plt.subplot(3, 2, 1)

x1 = [15, 19, 15, 5, 13, 17, 15, 12, 8, 6, 9, 13]
x2 = [13, 16, 17, 6, 17, 14, 15, 13, 7, 6, 10, 12]
plt.scatter(x1, x2)

X = np.array(list(zip(x1, x2)))

c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
m = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

p = 1
for i in [2, 3, 4, 5, 8]:
	p += 1
	plt.subplot(3, 2, p)
	model = KMeans(n_clusters=i).fit(X)
	print (model.labels_)
	for i, j in enumerate(model.labels_):
	    plt.plot(x1[i], x2[i], color=c[j], marker=m[j],ls='None')
	print (metrics.silhouette_score(X, model.labels_ ,metric='euclidean'))
plt.show()
