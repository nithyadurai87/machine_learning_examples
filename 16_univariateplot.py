import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("14_input_data.csv")
df = df.fillna(0)
df = df[:100]

y = [i for i in range(0,10)]
fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111)
ax.set(title="Total Living Sq.Ft",
      ylabel='No of Houses', xlabel='Living Sq.Ft')
ax.hist(df['GrLivArea'])
plt.savefig('Histogram.jpg')

sns.distplot(df['GrLivArea'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
plt.savefig('DensityPlot.jpg')

fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111)
ax.set(title="Total Living Sq.Ft",
      ylabel='No of Houses', xlabel='Living Sq.Ft')
ax.boxplot(df['GrLivArea'])
plt.savefig('BoxPlot.jpg')

https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
