import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

m = len(y) 

theta0 = 1.5
theta1 = 0
predicted_y = [theta0+(theta1*1), theta0+(theta1*2), theta0+(theta1*3)]

sum=0
for i,j in zip(predicted_y,y):
  sum = sum+((i-j)**2) 
J = 1/(2*m)*sum  
print ("cost when theta0=1.5 theta1=0 :",J)

theta0a = 0
theta1a = 0.5
predicted_ya = [theta0a+(theta1a*1), theta0a+(theta1a*2), theta0a+(theta1a*3)]

suma=0
for i,j in zip(predicted_ya,y):
  suma = suma+((i-j)**2) 
J = 1/(2*m)*suma  
print ("cost when theta0=0 theta1=0.5 :",J)

theta0b = 1
theta1b = 0.5
predicted_yb = [theta0b+(theta1b*1), theta0b+(theta1b*2), theta0b+(theta1b*3)]

sumb=0
for i,j in zip(predicted_yb,y):
  sumb = sumb+((i-j)**2) 
J = 1/(2*m)*sumb  
print ("cost when theta0=1 theta1=0.5 :",J)
