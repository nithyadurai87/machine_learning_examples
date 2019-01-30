import matplotlib.pyplot as plt
import pandas as pd
import os.path
import os.path as osp
import pickle
from sklearn.externals import joblib
import glob
from sklearn.model_selection import train_test_split

def Sensitivity_Analysis(df,txt):
	mark_list = ['d']
	model = joblib.load("./salepriceprediction.pkl")
	d = {}
	min_variable = 334
	max_variable = 5642
	step = 500
	for i in range(0,len(df)):
		df2=pd.DataFrame(df.iloc[i])
		actual=df.iloc[i]['GrLivArea']
		df2=df2.T
		df3=df2.copy()
		var_list=[]
		curval = actual.copy()
		while(curval <= max_variable):
			var_list.append(curval)
			curval = curval+step
		curval = actual
		while(curval >= min_variable):
			var_list.append(curval)
			curval = curval-step
		var_list.sort()
		var_list = pd.Series(var_list).drop_duplicates().reset_index(drop=True)
		num_point = len(var_list)
		for j in range(1,num_point):
			df2=df2.append(df3)
		df2.reset_index(drop=True,inplace=True)
		df2['GrLivArea']=var_list
		d[str(i)]=df2
		del df2     
      
	for key,value in d.items():
		df_sense=d[key]
		df_sense=pd.DataFrame(df_sense)
		predicted_Sense=model.predict(df_sense)
		d222={'Input':list(df_sense['GrLivArea']),'Output':predicted_Sense}
		df_sensitivity=pd.DataFrame(data=d222)
		df_sensitivity.set_index('Input',inplace=True)
		df_sensitivity.sort_index(inplace=True)
		plt.plot(list(df_sensitivity.index),list(df_sensitivity['Output']))
		plt.xlabel('GrLivArea')
		plt.ylabel('SalePrice')
		del predicted_Sense
		del df_sense
	plt.grid()   
	plt.show()
	plt.close()        
    
 
df = pd.read_csv("./23_sens_analysis.csv")
Sensitivity_Analysis(df,'1')