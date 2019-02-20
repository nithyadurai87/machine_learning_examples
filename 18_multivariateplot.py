import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import plotly
import plotly.graph_objs as go
import numpy as np

df = pd.read_csv("14_input_data.csv")
parallel_coordinates(df, 'SalePrice')
plt.savefig('ParallelCoordinates.jpg')

desc_data = df.describe()
desc_data.to_csv('./metrics.csv')

data = [
    go.Parcoords(
        line = dict(colorscale = 'Jet',
                   showscale = True,
                   reversescale = True,
                   cmin = -4000,
                   cmax = -100),
        dimensions = list([
            dict(range = [1,10],
                 label = 'OverallQual', values = df['OverallQual']),
            dict(range = [0,6110],
                 label = 'TotalBsmtSF', values = df['TotalBsmtSF']),
            dict(tickvals = [334,4692],
                 label = '1stFlrSF', values = df['1stFlrSF']),
            dict(range = [334,5642],                 
                 label = 'GrLivArea', values = df['GrLivArea']),
            dict(range = [0,3],
                 label = 'FullBath', values = df['FullBath']),
            dict(range = [2,14],
                 label = 'TotRmsAbvGrd', values = df['TotRmsAbvGrd']),
            dict(range = [0,3],
                 label = 'Fireplaces', values = df['Fireplaces']),
            dict(range = [0,4],
                 label = 'GarageCars', values = df['GarageCars']),
            dict(range = [0,1418],
                 label = 'GarageArea', values = df['GarageArea']),
            dict(range = [34900,555000],
                 label = 'SalePrice', values = df['SalePrice'])               
        ])
    )
]
plotly.offline.plot(data, filename = './parallel_coordinates_plot.html', auto_open= True)
