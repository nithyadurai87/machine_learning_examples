import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import os

df = pd.read_csv('/content/sample_data/06_output_data.csv')

i = list(df.columns.values)
i.pop(i.index('SalePrice'))
df0 = df[i+['SalePrice']]
df = df0.select_dtypes(include=['integer','float'])

X = df[list(df.columns)[:-1]]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y)

def linear():
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def ridge():
  scaler = StandardScaler()
  X_normalized = scaler.fit_transform(X_train)
  ridge = Ridge()
  ridge.fit(X_normalized, y_train)
  y_predictions = ridge.predict(X_test)
  return (ridge.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def lasso():
  scaler = StandardScaler()
  X_normalized = scaler.fit_transform(X_train)
  lasso = Lasso()
  lasso.fit(X_normalized, y_train)
  y_predictions = lasso.predict(X_test)
  return (lasso.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def elasticnet():
  scaler = StandardScaler()
  X_normalized = scaler.fit_transform(X_train)
  elasticNet = ElasticNet()
  elasticNet.fit(X_normalized, y_train)
  y_predictions = elasticNet.predict(X_test)
  return (elasticNet.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def randomforest():
	regressor = RandomForestRegressor(n_estimators=15,min_samples_split=15,criterion='friedman_mse',max_depth=None)
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	print("Selected Features for RamdomForest",regressor.feature_importances_)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def perceptron():
	regressor = MLPRegressor(hidden_layer_sizes=(5000,), activation='relu', solver='adam', max_iter=1000)
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	print("Co-efficients of Perceptron",regressor.coefs_)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))
	
def decisiontree():
	regressor = DecisionTreeRegressor(min_samples_split=30,max_depth=None)
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	print("Selected Features for DecisionTrees",regressor.feature_importances_)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def adaboost():
	regressor = AdaBoostRegressor(random_state=8, loss='exponential').fit(X_train, y_train)
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	print("Selected Features for Adaboost",regressor.feature_importances_)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))
	
def extratrees():
	regressor = ExtraTreesRegressor(n_estimators=50).fit(X_train, y_train)
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	print("Selected Features for Extratrees",regressor.feature_importances_)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

def gradientboosting():
	regressor = GradientBoostingRegressor(loss='quantile',n_estimators=500, min_samples_split=15).fit(X_train, y_train)
	regressor.fit(X_train, y_train)
	y_predictions = regressor.predict(X_test)
	print("Selected Features for Gradientboosting",regressor.feature_importances_)
	return (regressor.score(X_test, y_test),sqrt(mean_squared_error(y_test, y_predictions)))

print ("Score, RMSE values")
print ("Linear = ",linear())
print ("Ridge = ",ridge())
print ("Lasso = ",lasso())
print ("ElasticNet = ",elasticnet())
print ("RandomForest = ",randomforest())
print ("Perceptron = ",perceptron())
print ("DecisionTree = ",decisiontree())
print ("AdaBoost = ",adaboost())
print ("ExtraTrees = ",extratrees())
print ("GradientBoosting = ",gradientboosting())
