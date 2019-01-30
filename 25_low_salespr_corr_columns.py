import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('23_input_data.csv', sep = ',')

cols = df.columns
slopelist = []	

dfs = df[cols[-1]].tolist()
dfs = [int(i) if int(i)<250000 else None for i in dfs]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([i for i in range(1460)],dfs , marker="o", color="red", zorder=10)
ax.plot([i for i in range(1460)],df[cols[-1]], marker='o', zorder=0)
ax.set(title="sales price below 250000(Red)")
ax.legend()
plt.savefig('lowsalesprice.jpg')
plt.show()

for i in df.columns[:-1]:

    name = i + ".jpg"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(1460)],df[i] , marker="x", color="blue", zorder=10)
    plt.savefig(name)
    plt.show()