import pandas as pd 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 
import numpy as np 
data = pd.read_csv("python-ml-course-master/datasets/ads/Advertising.csv")
lm = smf.ols(formula = "Sales~TV", data = data).fit()
print(lm.params)
print(lm.pvalues)
print(lm.rsquared)
lm2 = smf.ols(formula = "Sales~TV+Newspaper",data = data).fit()
print(lm2.params)
print(lm2.pvalues)

'''ecuacion : 5.7749 + 0.0469*TV + 0.0442*Newspaper'''

print(lm2.rsquared)
sales_pred = lm2.predict(data[["TV", "Newspaper"]])
print (sales_pred[0:10])
SSD = sum((data["Sales"]- sales_pred)**2)
RSE = np.sqrt(SSD/(len(data)-3))
sales_m = np.mean(data["Sales"])
print((RSE/sales_m)*100)

#Conclusion Newspaper no nos aporto nada:(
lm3 = smf.ols(formula = "Sales~TV+Radio", data = data).fit()
print(lm3.summary())
sales_pred = lm3.predict(data[["TV", "Radio"]])
SSD = sum((data["Sales"]- sales_pred)**2)
RSE = np.sqrt(SSD/(len(data)-3))
print((RSE/sales_m)*100)
lm4 = smf.ols(formula = "Sales~TV+Radio+Newspaper", data = data).fit()
print(lm4.summary())
sales_pred = lm4.predict(data[["TV", "Radio","Newspaper"]])
SSD = sum((data["Sales"]- sales_pred)**2)
RSE = np.sqrt(SSD/(len(data)-4))
print((RSE/sales_m)*100)

'''Multicolinealidad

nos da problemas Newspaper ~ TV + Radio -> R2   VIF ( inflacion) = 1/(1-R2)
TV ~ Newspaper+ Radio -> R2   VIF ( inflacion) = 1/(1-R2)
Radio ~ TV + Newspaper -> R2   VIF ( inflacion) = 1/(1-R2)

VIF > 5 = variable que debe ser eliminada (mucha correlacion entre variables predictoras)
'''
lm_news = smf.ols(formula = "Newspaper~TV+Radio", data = data).fit()
rsq_n = lm_news.rsquared
VIF = 1/(1-rsq_n)
print(VIF)
lm_tv = smf.ols(formula = "TV~Newspaper+Radio", data=data).fit()
rsq_tv = lm_tv.rsquared
VIF = 1/(1-rsq_tv)
print(VIF)
lm_rad = smf.ols(formula = "Radio~Newspaper+TV", data=data).fit()
rsq_rad = lm_rad.rsquared
VIF = 1/(1-rsq_rad)
print(VIF)