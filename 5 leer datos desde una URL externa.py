#Carga de datos a través de la función open de python para datasets excesivamente grandes
#Crea un for que lee linea por linea del fichero y elimina aquellas que termina de processar para no quedarse sin memoria
import pandas as pd
import csv
import urllib3

medals_url = "http://winterolympicsmedals.com/medals.csv"
data = pd.read_csv(medals_url)
#print(data.head())
http = urllib3.PoolManager()
r = http.request("GET", medals_url)
print(r.status)
print(r.data)
