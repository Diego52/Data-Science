#Carga de datos a través de la función read_csv

import pandas as pd 
import os

mainpath = "C:/Developer/Data science y machine learning course/python-ml-course-master/datasets"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)

data = pd.read_csv(fullpath)
#print(data)
data = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt")
#print(data)
#print(data.columns.values)
data_cols = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Columns.csv")
data_col_list = data_cols["Column_Names"].tolist()
#print(data_col_list)
data = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", header= None, names = data_col_list)
print(data.columns.values)