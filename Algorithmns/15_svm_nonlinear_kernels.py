import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.linear_model.logistic import LogisticRegression

df = pd.read_csv('./flowers.csv')
X = df[list(df.columns)[:-1]]
y = df['Flower']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)  
y_pred = logistic.predict(X_test) 
print ('Accuracy-logistic:', accuracy_score(y_test, y_pred))

gaussian = SVC(kernel='rbf') 
gaussian.fit(X_train, y_train)  
y_pred = gaussian.predict(X_test) 
print ('Accuracy-svm:', accuracy_score(y_test, y_pred))

