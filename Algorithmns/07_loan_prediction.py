import os 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv('./train1.csv')
l1 = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
l2 = ['Loan_Status']
l1_train, l1_test, l2_train, l2_test = train_test_split(data[l1], data[l2],test_size=0.25, random_state=1)

l2_train = l2_train.replace({'Y':1, 'N':0}).values
l2_test = l2_test.replace({'Y':1, 'N':0}).values     

print (l2_test)
