import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

plt.figure()
plt.title('Data - X and Y')
plt.plot(x,y,'*')
plt.xticks([0,1,2,3])
plt.yticks([0,1,2,3])
plt.show()

def linear_regression(theta0,theta1):
	predicted_y = []
	for i in x:
		predicted_y.append((theta0+(theta1*i)))
	plt.figure()
	plt.title('Predictions')
	plt.plot(x,predicted_y,'.')
	plt.xticks([0,1,2,3])
	plt.yticks([0,1,2,3])
	plt.show()	

theta0 = 1.5
theta1 = 0
linear_regression(theta0,theta1)

theta0a = 0
theta1a = 1.5
linear_regression(theta0a,theta1a)

theta0b = 1
theta1b = 0.5
linear_regression(theta0b,theta1b)

