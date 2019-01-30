import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error

def linear():
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

df = pd.read_csv('./14_input_data.csv')

X = df[list(df.columns)[:-1]]
y = df['SalePrice']

X = X[:600]
y = y[:600]

X_train, X_test, y_train, y_test = train_test_split(X, y)
print ("linear(Before Kfold) = ",linear())

i=X_train
j=y_train

for k_no, (x, y) in enumerate(KFold(n_splits=5,shuffle=True, random_state=42).split(i, j)):
	X_train = i.loc[i.index.intersection(x)]
	y_train = j.loc[j.index.intersection(x)] 
	X_test = i.loc[i.index.intersection(y)]
	y_test = j.loc[j.index.intersection(y)]
	print ("linear = ",linear())