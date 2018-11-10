import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import RFE
from sklearn.datasets import make_friedman1

df = pd.read_csv('./12_output_data.csv')

X = df[list(df.columns)[:-1]]
y = df['A']
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = DecisionTreeRegressor(min_samples_split=3,max_depth=None)
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print ("Selected Features for DecisionTree",regressor.feature_importances_)

# RFE Technique - Recursive Feature Elimination 
X, y = make_friedman1(n_samples=20, n_features=17, random_state=0)
selector = RFE(LinearRegression())
selector = selector.fit(X, y)
print ("Selected Features for LinearRegression",selector.ranking_)
