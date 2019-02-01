import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 

X = pd.DataFrame([100,200,300,400,500,600],columns=['sqft'])
y = pd.DataFrame([543543,34543543,35435345,34534,34534534,345345],columns=['Price'])

lin = LinearRegression()   
lin.fit(X, y) 
plt.scatter(X, y, color = 'blue')   
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('sqft') 
plt.ylabel('Price')   
plt.show() 

for i in [2,3,4,5]:
	poly = PolynomialFeatures(degree = i) 
	X_poly = poly.fit_transform(X)   
	poly.fit(X_poly, y) 
	lin2 = LinearRegression() 
	lin2.fit(X_poly, y) 
	plt.scatter(X, y, color = 'blue')  
	plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
	plt.title('Polynomial Regression') 
	plt.xlabel('sqft') 
	plt.ylabel('Price')   
	plt.show() 
		
 