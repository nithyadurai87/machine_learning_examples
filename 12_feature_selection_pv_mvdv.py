import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import RFE
from sklearn.datasets import make_friedman1

df = pd.read_csv('./11_output_data.csv')
print (df.shape,df.columns)

# Dropping columns which has correlation with target less than threshold
target = "A"
correlations = df.corr()[target].abs()
correlations = correlations.round(2)
correlations.to_csv('./12_PV_MVDV_correlation.csv',index=False)
df=df.drop(correlations[correlations<0.06].index, axis=1)

print (df.shape,df.columns)
df.to_csv('./12_output_data.csv',index=False)
