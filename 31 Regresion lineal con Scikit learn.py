from sklearn.feature_selection import RFE
from sklearn.svm import SVR 
import pandas as pd 
from sklearn.linear_model import LinearRegression
import numpy as np 

data = pd.read_csv("python-ml-course-master/datasets/ads/Advertising.csv")
features_cols = ["TV", "Radio", "Newspaper"]
x= data[features_cols]
y = data["Sales"]
estimator = SVR(kernel = "linear") #crea un modelo lineal
selector = RFE(estimator, 2, step=1) #Le pedimos que deje el modelo en 2 variables predictoras Recursive Feature Elimination
selector = selector.fit(x,y) 
print(selector.support_)
print(selector.ranking_)

X_pred = x[["TV","Radio"]] 
lm = LinearRegression() #Crea el modelo de regresion lineal
lm.fit(X_pred, y) #Ajusta el modelo a nuestros datos
print(lm.intercept_) #Alpha
print(lm.coef_) #Bethas 
print(lm.score(X_pred, y)) #R2