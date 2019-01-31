import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1, 800, 2, 15],[1, 1200, 3, 1],[1, 2400, 5, 5]])
y = [3000000,2000000,3500000]
theta = np.array([100, 1000, 10000, 100000])

print (x.dot(theta.transpose()))

