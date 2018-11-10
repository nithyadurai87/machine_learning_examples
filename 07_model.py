import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import os

df = pd.read_csv('./06_output_data.csv')

i = list(df.columns.values)
i.pop(i.index('SalePrice'))
df0 = df[i+['SalePrice']]
df = df0.select_dtypes(include=['integer','float'])
print (df.columns)

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

plt.plot(y_predictions,y_test,'r.') 
plt.plot(y_predictions,y_predictions,'k-') 
plt.title('Parity Plot - Linear Regression')
plt.show()

plot = plt.scatter(y_predictions, (y_predictions - y_test), c='b')
plt.hlines(y=0, xmin= 100000, xmax=400000)
plt.title('Residual Plot - Linear Regression')
plt.show()

joblib.dump(regressor, './07_output_salepricemodel.pkl')
