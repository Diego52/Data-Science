#resumen de los datos: dimensiones y estructuras
import pandas as pd 
import os

mainpath = "C:/Developer/Data science y machine learning course/python-ml-course-master/datasets"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)
urldata = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"


data = pd.read_csv(fullpath)

#print(pd.isnull(data["body"]))
#print(pd.notnull(data["body"]))
print(pd.isnull(data["body"]).values)
print(pd.isnull(data["body"]).values.ravel().sum())
'''
 Los valores que faltan en un data set pueden venir por dos razones:
    *Extraccion de los datos
    *Recoleccion de los datos
'''
#borrado de datos 
#print(data.dropna(axis=0, how='all'))
#remplazo de valores
data2 = data
#print(data2.fillna("Desconocido"))
data2["body"] = data2["body"].fillna(0)
data2["home.dest"] = data2["home.dest"].fillna("desconocido")
#print(data2.tail())
#data2["age"] = data2["age"].fillna(data["age"].mean())
data2["age"] = data2["age"].fillna(method="ffill")
print(data2["age"])