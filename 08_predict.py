import os 
import json
import pandas as pd
import numpy
from sklearn.externals import joblib

s = pd.read_json('./08_input.json')
p = joblib.load("./07_output_salepricemodel.pkl")
r = p.predict(s)
print (str(r))
