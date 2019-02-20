import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1, 800, 2, 15],[1, 1200, 3, 1],[1, 2400, 5, 5]])
y = np.array([3000000,2000000,3500000])
theta = np.array([100, 1000, 10000, 100000])

predicted_y = x.dot(theta.transpose())
print (predicted_y)

m = y.size
diff = predicted_y - y
squares = np.square(diff)
#sum_of_squares = 5424168464
sum_of_squares = np.sum(squares)
cost_fn = 1/(2*m)*sum_of_squares 
print (diff)
print (squares)
print (sum_of_squares)
print (cost_fn)

