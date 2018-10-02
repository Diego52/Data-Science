import pandas as pd 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 
import numpy as np 
data = pd.read_csv("python-ml-course-master/datasets/ads/Advertising.csv")

lm = smf.ols(formula = "Sales~TV", data = data).fit()

print(lm.params)
'''
intercept 7.032
Tv 0.047
dtype: float64

El modelo lineal seria Sales = 7.032 + 0.047 * TV

'''
print(lm.pvalues)

'''
intercept 1.406 e-35
tv 1.4673 e-42

son tan pequeños que podemos decir que si son significativos

'''
print(lm.rsquared, lm.rsquared_adj)
'''
0.6118
0.6099

'''
print(lm.summary())

sales_pred = lm.predict(pd.DataFrame(data["TV"]))

data.plot(kind = "scatter", x = "TV", y = "Sales")
plt.plot(pd.DataFrame(data["TV"]), sales_pred, "r", 2)
plt.show()

data["sales_pred"] = sales_pred
data["RSE"] = (data["Sales"]-data["sales_pred"]) **2
SSD = sum(data["RSE"])
RSE = np.sqrt(SSD/(len(data)-2))
print(SSD,RSE)
print(data.head())
sales_m = np.mean(data["Sales"])
print(sales_m)
error = (RSE/sales_m )*100
print(error)
plt.hist((data["Sales"]-data["sales_pred"]))
plt.show()