#resumen de los datos: dimensiones y estructuras
import pandas as pd 
import os

mainpath = "C:/Developer/Data science y machine learning course/python-ml-course-master/datasets"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)
urldata = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"


data = pd.read_csv(fullpath)
#print(data.head(10))
#print(data.tail(10))
#print(data.columns[0])
#print(data.columns.values[0])

#Resumen de los estadisticos basicos de las variables numericas
print(data.describe())
print(data.dtypes)
print(data.shape)