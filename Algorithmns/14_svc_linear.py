import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression

def classifier():
    xx = np.linspace(1,10)
    yy = -regressor.coef_[0][0] / regressor.coef_[0][1] * xx - regressor.intercept_[0] / regressor.coef_[0][1]
    plt.plot(xx, yy)
    plt.scatter(x1,x2)
    plt.show()

x1 = [2,6,3,9,4,10]
x2 = [3,9,3,10,2,13]

X = np.array([[2,3],[6,9],[3,3],[9,10],[4,2],[10,13]])
y = [0,1,0,1,0,1]

regressor = LogisticRegression()
regressor.fit(X,y)
classifier()

regressor = svm.SVC(kernel='linear',C = 1.0)
regressor.fit(X,y)
classifier()


#http://mlwiki.org/index.php/Support_Vector_Machines
#https://medium.com/data-py-blog/kernel-svm-in-python-a8fae37908b9
