import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

df = pd.read_csv("python-ml-course-master/datasets/ecom-expense/Ecom Expense.csv")

dummy_gender = pd.get_dummies(df["Gender"], prefix="Gender")
dummy_city_tier = pd.get_dummies(df["City Tier"], prefix="City")
print(dummy_gender.head())
print(dummy_city_tier.head())
column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
print(df_new.head())
df_new = df_new[column_names].join(dummy_city_tier)
print(df_new.head())
feature_cols = ["Monthly Income", "Transaction Time", "Gender_Female", "Gender_Male", "City_Tier 1", 
    "City_Tier 2","City_Tier 3","Record"]
x = df_new[feature_cols]
y = df_new["Total Spend"]
lm = LinearRegression()
lm.fit(x,y)

print(lm.intercept_)
#print(lm.coef_)
print(list(zip(feature_cols, lm.coef_))) #los junta

print(lm.score(x,y)) #R2

'''
El modelo puede ser escrito como:
Total_spend = -79.417 + Monthly Income * 0.14 + ....

'''

df_new["pred"] = lm.predict(df_new[feature_cols])
print(df_new.head())
SSD = np.sum((df_new["pred"] - df_new["Total Spend"]) **2)
RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1))
sales_mean = np.mean(df_new["Total Spend"])
error = (RSE/sales_mean)*100
print(error)

#Eliminar variables dummys redundantes 

dummy_gender = pd.get_dummies(df["Gender"], prefix="Gender").iloc[:,1:]
dummy_city_tier = pd.get_dummies(df["City Tier"], prefix = "City").iloc[:,1:]
column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new = df_new[column_names].join(dummy_city_tier)

feature_cols = ["Monthly Income", "Transaction Time", "Gender_Male",
    "City_Tier 2","City_Tier 3","Record"]
x = df_new[feature_cols]
y = df_new["Total Spend"]
lm = LinearRegression()
lm.fit(x,y)

print(lm.intercept_)
#print(lm.coef_)
print(list(zip(feature_cols, lm.coef_))) #los junta

print(lm.score(x,y)) #R2
df_new["pred"] = lm.predict(df_new[feature_cols])
print(df_new.head())
SSD = np.sum((df_new["pred"] - df_new["Total Spend"]) **2)
RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1))
sales_mean = np.mean(df_new["Total Spend"])
error = (RSE/sales_mean)*100
print(error)
