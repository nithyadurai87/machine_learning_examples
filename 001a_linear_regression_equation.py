import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]
plt.figure()
plt.title('Data - X and Y')
plt.plot(x,y,'*')
plt.xticks([0,1,2,3])
plt.yticks([0,1,2,3])
plt.show()


theta0 = 1.5
theta1 = 0
predicted_y = [theta0+(theta1*1), theta0+(theta1*2), theta0+(theta1*3)]
plt.figure()
plt.title('Data - X and Y with theta0 = 1.5 theta1 = 0')
plt.plot(x,predicted_y,'.')
plt.xticks([0,1,2,3])
plt.yticks([0,1,2,3])
plt.show()

theta0a = 0
theta1a = 0.5
predicted_ya = [theta0a+(theta1a*1), theta0a+(theta1a*2), theta0a+(theta1a*3)]
plt.figure()
plt.title('Data - X and Y with theta0 = 0 theta1 = 0.5')
plt.plot(x,predicted_ya,'.')
plt.xticks([0,1,2,3])
plt.yticks([0,1,2,3])
plt.show()

theta0b = 1
theta1b = 0.5
predicted_yb = [theta0b+(theta1b*1), theta0b+(theta1b*2), theta0b+(theta1b*3)]
plt.figure()
plt.title('Data - X and Y with theta0 = 1 theta1 = 0.5')
plt.plot(x,predicted_yb,'.')
plt.xticks([0,1,2,3])
plt.yticks([0,1,2,3])
plt.show()