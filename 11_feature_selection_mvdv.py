import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import RFE
from sklearn.datasets import make_friedman1

df = pd.read_csv('./11_input_data.csv')

# Dropping all process parameters
df = df.drop(["A","B", "C", "D", "E", "F"], axis=1) 

#finding correlation between manipulated & disturbance variables
correlations = df.corr()
correlations = correlations.round(2)
correlations.to_csv('11_MV_DV_correlation.csv',index=False)
fig = plt.figure()
g = fig.add_subplot(111)
cax = g.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,20,1)
g.set_xticks(ticks)
g.set_yticks(ticks)
g.set_xticklabels(list(df.columns))
g.set_yticklabels(list(df.columns))
plt.savefig('11_MV_DV_correlation.png')

#removing parameters with high correlation 
upper = correlations.where(numpy.triu(numpy.ones(correlations.shape), k=1).astype(numpy.bool))
cols_to_drop = []
for i in upper.columns:
	if (any(upper[i] == -1) or any(upper[i] == -0.98) or any(upper[i] == -0.99) or any(upper[i] == 0.98) or any(upper[i] == 0.99) or any(upper[i] == 1)):
		cols_to_drop.append(i)
df = df.drop(cols_to_drop, axis=1) 

print (df.shape,df.columns)
df.to_csv('./11_output_data.csv',index=False)
