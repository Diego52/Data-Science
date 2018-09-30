#Carga de datos a través de la función open de python para datasets excesivamente grandes
#Crea un for que lee linea por linea del fichero y elimina aquellas que termina de processar para no quedarse sin memoria
import pandas as pd

mainpath = "C:/Developer/Data science y machine learning course/python-ml-course-master/datasets"
filepath = "titanic/titanic3.xls"

titanic = pd.read_excel(mainpath + "/" + filepath, "titanic3")
print(titanic.head())
filepath = "titanic/titanic3.xlsx"
titanic = pd.read_excel(mainpath + "/" + filepath, "titanic3")
print(titanic.head())

titanic.to_csv(mainpath + "/titanic/titanic3_custom.csv")
titanic.to_excel(mainpath + "/titanic/titanic3_custom.xls")
titanic.to_json(mainpath + "/titanic/titanic3_custom.json")
