
x = [1, 2, 3]
y = [1, 2, 3]

m = len(y)

theta0 = 1
theta1 = 1.5
alpha = 0.01

def cost_function(theta0,theta1):
	predicted_y = [theta0+(theta1*1), theta0+(theta1*2), theta0+(theta1*3)]
	sum=0
	for i,j in zip(predicted_y,y):
		sum = sum+((i-j)**2) 
	J = 1/(2*m)*sum 
	return (J)

def gradientDescent(x, y, theta1, alpha):
	J_history = []
	for i in range(50):
		for i,j in zip(x,y):
			delta=1/m*(i*i*theta1-i*j);
			theta1=theta1-alpha*delta;
			J_history.append(cost_function(theta0,theta1))
	print (min(J_history))
	

gradientDescent(x, y, theta1, alpha)