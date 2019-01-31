import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 800, 2, 15])
y = [3000000]
theta = np.array([100, 1000, 10000, 100000])

print (np.matmul(theta.transpose(),x))

