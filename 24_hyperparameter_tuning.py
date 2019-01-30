import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'./23_input_data.csv')

X = df[list(df.columns)[:-1]]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = ExtraTreesRegressor()

extra_grid = {'n_estimators': [5,10,20,50,100,150,200],
'max_features' : ['auto', 'sqrt'],
'min_samples_leaf' : [2,5,6,10],
'max_depth' : [2,6,7,5]
}

# 500, 6 7 8, 3 4

Grid_search = GridSearchCV(regressor, param_grid = extra_grid)
Grid_search.fit(X_train, y_train)
Model_var = Grid_search.best_estimator_
y_predictions = Model_var.predict(X_test)
meanSquaredError=mean_squared_error(y_test, y_predictions)
rootMeanSquaredError = sqrt(meanSquaredError)
	
print("Number of predictions:",len(y_predictions))
print("Mean Squared Error:", meanSquaredError)
print("Root Mean Squared Error:", rootMeanSquaredError)
print ("Scoring:",Model_var.score(X_test, y_test))

