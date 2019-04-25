import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("14_input_data.csv")
df = df.fillna(0)
df = df[:100]

#Histogram
y = [i for i in range(0,10)]
fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111)
ax.set(title="Histogram",
      ylabel='No of Houses', xlabel='Living Sq.Ft')
ax.hist(df['GrLivArea'])
plt.show()

#DensityPlot
sns.distplot(df['GrLivArea'], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3})
plt.show()

#BoxPlot
fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111)
ax.set(title="Box Plot",
      ylabel='No of Houses', xlabel='Living Sq.Ft')
ax.boxplot(df['GrLivArea'])
plt.show()
