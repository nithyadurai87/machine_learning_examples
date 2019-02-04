from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(figsize=(8, 5), 
                        subplot_kw={'projection': '3d'})					
						

values = 2
r = np.linspace(-values,values,100)

theta0,theta1= np.meshgrid(r,r)

original_y = [1, 2, 3]
m = len(original_y) 

predicted_y = [theta0+(theta1*1), theta0+(theta1*2), theta0+(theta1*3)]

sum=0
for i,j in zip(predicted_y,original_y):
  sum = sum+((i-j)**2) 
J = 1/(2*m)*sum  

ax1.plot_wireframe(theta0,theta1,J)
ax1.set_title("plot")

plt.show()