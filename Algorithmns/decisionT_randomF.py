import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.model_selection import train_test_split 

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=8, centers=2,
                  random_state=0, cluster_std=1.0)
print (X)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');
plt.show()


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)
visualize_classifier(tree, X, y)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,random_state=1).fit(X, y)
visualize_classifier(bag, X, y)

from sklearn.ensemble import RandomForestClassifier
rdm = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
visualize_classifier(rdm, X, y);

"""
















https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html
https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
"""

