import pandas as pd

# data can be downloaded from the url: https://www.kaggle.com/vikrishnan/boston-house-prices 
df = pd.read_csv('./06_input_data.csv')

# Understanding data
print (df.shape)
print (df.columns)
print(df.head(5))
print(df.info())
print(df.describe())
print(df.groupby('LotShape').size())

# Dropping null value columns which cross the threshold
a = df.isnull().sum()
print (a)
b =  a[a>(0.05*len(a))]
print (b)
df = df.drop(b.index, axis=1)
print (df.shape)

# Replacing null value columns (text) with most used value
a1 = df.select_dtypes(include=['object']).isnull().sum()
print (a1)
print (a1.index)
for i in a1.index:
	b1 = df[i].value_counts().index.tolist()
	print (b1)
	df[i] = df[i].fillna(b1[0])
	
# Replacing null value columns (int, float) with most used value
a2 = df.select_dtypes(include=['integer','float']).isnull().sum()
print (a2)
b2 = a2[a2!=0].index 
print (b2)
df = df.fillna(df[b2].mode().to_dict(orient='records')[0])

# Creating new columns from existing columns
print (df.shape)
a3 = df['YrSold'] - df['YearBuilt']
b3 = df['YrSold'] - df['YearRemodAdd']
df['Years Before Sale'] = a3
df['Years Since Remod'] = b3
print (df.shape)

# Dropping unwanted columns
df = df.drop(["Id", "MoSold", "SaleCondition", "SaleType", "YearBuilt", "YearRemodAdd"], axis=1) 
print (df.shape)

# Dropping columns which has correlation with target less than threshold
target='SalePrice'
x = df.select_dtypes(include=['integer','float']).corr()[target].abs()
print (x)  
df=df.drop(x[x<0.4].index, axis=1)
print (df.shape)

# Checking for the necessary features after dropping some columns
l1 = ["PID","MS SubClass","MS Zoning","Street","Alley","Land Contour","Lot Config","Neighborhood","Condition 1","Condition 2","Bldg Type","House Style","Roof Style","Roof Matl","Exterior 1st","Exterior 2nd","Mas Vnr Type","Foundation","Heating","Central Air","Garage Type","Misc Feature","Sale Type","Sale Condition"]
l2 = []
for i in l1:
    if i in df.columns:
        l2.append(i)
       
# Getting rid of nominal columns with too many unique values
for i in l2:
    len(df[i].unique())>10
    df=df.drop(i, axis=1)
print (df.columns)

df.to_csv('06_output_data.csv',index=False)
