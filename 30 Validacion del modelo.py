import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
#Dividir el dataset en conjunto de entrenamiento y de testing

data = pd.read_csv("python-ml-course-master/datasets/ads/Advertising.csv")

a = np.random.randn(len(data))
check = (a<0.8)
training = data[check]
testing = data[~check]

lm = smf.ols(formula = "Sales~TV+Radio", data= training).fit()
print(lm.summary())

#Validacion del modelo
sales_pred = lm.predict(testing)
SSD = sum((testing["Sales"]-sales_pred)**2)
RSE = np.sqrt(SSD / (len(testing)-3))
sales_mean = np.mean(testing["Sales"])
error = RSE / sales_mean
print(error)