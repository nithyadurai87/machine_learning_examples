import pandas as pd
import pylab
import numpy as np
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib._pylab_helpers

df = pd.read_csv('./14_input_data.csv')

# Finding outlier in data
for i in range(len(df.columns)):
    pylab.figure()
    pylab.boxplot(df[df.columns[i]])
    #pylab.violinplot(df[df.columns[i]])
    pylab.title(df[df.columns[i]].name)

list1=[]

for i in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
    list1.append(i.canvas.figure)
print (list1)
    
for i, j in enumerate(list1):
    j.savefig(df[df.columns[i]].name)

# Removing outliers
z = np.abs(stats.zscore(df))
print(z)
print(np.where(z > 3))
print(z[53][9])
df1 = df[(z < 3).all(axis=1)]
print (df.shape)
print (df1.shape)