import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1, 800, 2, 15],[1, 1200, 3, 1],[1, 2400, 5, 5]])
y = np.array([3000000,2000000,3500000])
theta = np.array([100, 1000, 10000, 100000])

predicted_y = x.dot(theta.transpose())

diff = predicted_y - y
print (np.square(diff))

"""

m = y.size
sum=0
for i,j in zip(np.nditer(predicted_y),np.nditer(y)):
	print (i-j)
	print (type(i-j))
	print (np.array((i-j), dtype='int32')**2)
"""
	
