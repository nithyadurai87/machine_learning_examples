import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

df = pd.read_csv(r'./23_input_data.csv')
df1 = pd.read_csv(r'./23_sens_analysis.csv')

X = df[list(df.columns)[:-1]]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = ExtraTreesRegressor(n_estimators=50)
regressor.fit(X_train, y_train)
joblib.dump(regressor, r'./salepriceprediction.pkl')
lin_reg = joblib.load(r"./salepriceprediction.pkl")
y_predictions = regressor.predict(X_test)
meanSquaredError=mean_squared_error(y_test, y_predictions)
rootMeanSquaredError = sqrt(meanSquaredError)

print("Number of predictions:",len(y_predictions))
print("Mean Squared Error:", meanSquaredError)
print("Root Mean Squared Error:", rootMeanSquaredError)
print ("Scoring:",regressor.score(X_test, y_test))

result = lin_reg.predict(df1)
print (result)