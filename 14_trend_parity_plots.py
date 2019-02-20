import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import os

df = pd.read_csv('./14_input_data.csv')

X = df[list(df.columns)[:-1]]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predictions = regressor.predict(X_test)

meanSquaredError=mean_squared_error(y_test, y_predictions)
rootMeanSquaredError = sqrt(meanSquaredError)

print("Number of predictions:",len(y_predictions))
print("Mean Squared Error:", meanSquaredError)
print("Root Mean Squared Error:", rootMeanSquaredError)
print ("Scoring:",regressor.score(X_test, y_test))

## TREND PLOT
y_test25 = y_test[:35]
y_predictions25 = y_predictions[:35]
myrange = [i for i in range(1,36)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
plt.plot(myrange,y_test25, marker='o')
plt.plot(myrange,y_predictions25, marker='o') 
plt.title('Trend between Actual and Predicted - 35 samples')
ax.set_xlabel("No. of Data Points")
ax.set_ylabel("Values- SalePrice")
plt.legend(['Actual points','Predicted values'])
plt.savefig('TrendActualvsPredicted.png',dpi=100)
plt.show()


## PARITY PLOT
y_testp = y_test[:]+50000
y_testm = y_test[:]-50000
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
plt.plot(y_test,y_predictions,'r.') 
plt.plot(y_test,y_test,'k-',color = 'green')
plt.plot(y_test,y_testp,color = 'blue')
plt.plot(y_test,y_testm,color = 'blue')
plt.title('Parity Plot')
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
plt.legend(['Actual vs Predicted points','Actual value line','Threshold of 50000'])
plt.show()

## Data Distribution
fig = plt.figure()
plt.plot([i for i in range(1,1461)],y,'r.')
plt.title('Data Distribution')
plt.show()

a, b = 0 , 0
for i in range(0,1460):
    if(y[i]>250000):
        a += 1
    else: 
        b +=1
print(a, b)

#X = X[:600]
#y = y[:600]


