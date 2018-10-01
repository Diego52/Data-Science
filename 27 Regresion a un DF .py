import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

'''
y = a + b * x
b = sum((xi - x_m) * (y_i - y_m)) / sum((xi - x_m) **2)
a = y_m - b * x_m

'''
x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100) 
y_act = 5 + 1.9 * x + res
x_list = x.tolist()
y_act_list = y_act.tolist()
data = pd.DataFrame(
    {
        "x":x_list, 
        "y_actual":y_act_list
    }
)
x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])

data["b_n"] = (data["x"] - x_mean) * (data["y_actual"] - y_mean)
data["b_d"] = (data["x"] - x_mean) **2
beta = sum(data["b_n"]) / sum(data["b_d"])
alpha = y_mean - beta * x_mean

data["y_model"] = alpha + beta * data["x"]
SSR = sum((data["y_model"] - y_mean)**2)
SSD = sum((data["y_model"] - data["y_actual"]) **2)
SST = sum((data["y_actual"] - y_mean) **2) 
print ('SSR',SSR,'SSD', SSD,'SST', SST)
R2 = SSR / SST
print('R2',R2)
RSE = np.sqrt(SSD/(len(data)-2))
print('RSE', RSE)
porcentaje = (RSE / y_mean) *100
print('Porcentaje de error', porcentaje, '%')
y_mean = [np.mean(y_act) for i in range(1,len(x_list) + 1)]
plt.plot(data["x"], data["y_actual"], "ro")
plt.plot(data["x"], data["y_model"], "r")
plt.plot(data["x"], y_mean, "g")
plt.title("Valor actual vs prediccion")
plt.show()
print(data.head())