from sklearn.linear_model import LinearRegression
from numpy.linalg import lstsq

x = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(x,y)

x1 = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y1 = [[11], [8.5], [15], [18], [11]]

predictions = model.predict(x1)
for i, prediction in enumerate(predictions):
	print ((prediction, y1[i]))
	
print (lstsq(x, y, rcond=None)[0])
	
print ('R-squared score = ',model.score(x1, y1))



