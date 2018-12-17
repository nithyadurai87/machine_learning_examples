import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("14_input_data.csv")
df = df.fillna(0)
df = df[:500]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.set(title='Living area vs Price of the house',
      xlabel='Price', ylabel='Area')
price = df['SalePrice'].tolist()
area = df['GrLivArea'].tolist()
ax.scatter(price,area)
plt.savefig('ScatterPlot.jpg')

df2 = pd.DataFrame()
df2['sale'] = df['SalePrice']
df2['area'] = df['GrLivArea']
fig = plt.figure(figsize=(12,12))
r = sns.heatmap(df2, cmap='BuPu')
plt.savefig('HeatMapSeaborn.jpg')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.set(title="Total Living Sq.Ft",
      ylabel='No of Houses', xlabel='Living Sq.Ft')
ax.hist2d(price,area,bins=100)
plt.savefig('HeatMapMatplotlib.jpg')
