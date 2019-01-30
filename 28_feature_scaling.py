import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'./23_input_data.csv')
X = df[list(df.columns)[:-1]]

print (X)

print (preprocessing.scale(X))

scaler = StandardScaler()
print(scaler.fit(X))
print(scaler.transform(X))