import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")
plt.hist(data["Day Calls"], bins= int(1 + np.log2(3333))) #Regla de Sturges de estadistica, para las div de un histograma
#Bins puede ser [0,30,60,..]
plt.xlabel("Numero de llamadas al dia")
plt.ylabel("Frecuencia")
plt.title("Histograma de llamadas al dia")
plt.show()
