import pandas as pd 
import os

mainpath = "C:/Developer/Data science y machine learning course/python-ml-course-master/datasets"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)
urldata = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"
data = pd.read_csv(fullpath)
def createDummy(df, var_name): 
    dummy = pd.get_dummies(df[var_name], prefix = var_name)
    df = df.drop(var_name,axis=1)
    #colum_name = df.columns.values.tolist()
    df = pd.concat([data,dummy],axis=1)
    return df
print(createDummy(data,"sex"))