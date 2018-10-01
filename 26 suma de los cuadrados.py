'''
y = a + b * x
X : 100 valores distribuidos segun una N(1.5,2.5) #normal de media 1.5 y desviacion estandar 2.5
Ye = 5 + 1.9 * x + e
e estara distribuido segun una N(0,0.8)
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
x = 1.5 + 2.5 * np.random.randn(100)

res = 0 + 0.8 * np.random.randn(100)

y_pred = 5 + 1.9 * x 

y_act = 5 + 1.9 * x + res

x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()
y_mean = [np.mean(y_act) for i in range(1,len(x_list) + 1)]
data = pd.DataFrame(
    {
        "x":x_list, 
        "y_actual":y_act_list,
        "y_prediccion":y_pred_list
    }
)
data["SSR"] = (data["y_prediccion"] - np.mean(y_act)) **2 #Error que explica el modelo
data["SSD"] = (data["y_prediccion"]-data["y_actual"]) **2 #Error que no explica el modelo
data["SST"] = (data["y_actual"]-np.mean(y_act)) **2 #Error total, se espera sea cercano a SSR
SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])
R2 = SSR / SST #Entre mas cercano a 1 mejor
print(R2,SSR,SSD,SST)
print(data.head())
plt.plot(x,y_pred)
plt.plot(x,y_act, "ro")
plt.plot(x,y_mean,"g")
plt.title("Valor actual vs prediccion")
plt.show()
plt.hist((data["y_prediccion"]-data["y_actual"]))
plt.show()