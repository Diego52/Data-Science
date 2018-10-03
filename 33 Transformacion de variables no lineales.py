import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def model_validation(x,y,y_predict):
    print("R2:",lm.score(x,y))
    print("Alpha:",lm.intercept_)
    print("Bethas:",lm.coef_)
    SSD = np.sum((y - y_predict)**2)
    RSE = np.sqrt(SSD/(len(x)-1))
    y_mean = np.mean(y)
    error = (RSE/y_mean) * 100
    print("Error%:", error)
data = pd.read_csv("python-ml-course-master/datasets/auto/auto-mpg.csv")

data["mpg"] = data["mpg"].dropna()
data["horsepower"] = data["horsepower"].dropna()
plt.plot(data["horsepower"], data["mpg"], "ro")
plt.xlabel("Caballos de potencia")
plt.ylabel("Consumo (millas por galon")
plt.title("CV vs MPG")

#Modelo de regresion lineal
#mpg = a + b * horsepower

x = data["horsepower"].fillna(data["horsepower"].mean())
y = data["mpg"].fillna(data["mpg"].mean())
x = x[:, np.newaxis]
lm = LinearRegression()

lm.fit(x,y)

plt.plot(x,y,"ro")
plt.plot(x, lm.predict(x),color="blue")
plt.show()

y_pred = lm.predict(x)
model_validation(x,y,y_pred)

#Modelo de regresion cuadratico 
#mpg = a + b * horsepower**2

x = x**2
lm = LinearRegression()
lm.fit(x,y)
y_pred = lm.predict(x)
model_validation(x,y,y_pred)
#Modelo de regresion lineal y cuadratico 
#mpg = a + b * horsepower + c * horsepower**2
poly = PolynomialFeatures(degree=2)
x = poly.fit_transform(x)
lm = LinearRegression()
lm.fit(x, y)
y_pred = lm.predict(x)
model_validation(x,y,y_pred)
#modelo es 55.026hp + 0.001126hp**2

'''
for d in range (2,6):
    print("Regresion de grado " + str(d))
    poly = PolynomialFeatures(degree=d)
    x = poly.fit_transform(x)
    lm = LinearRegression()
    lm.fit(x, y)
    y_pred = lm.predict(x)
    model_validation(x,y,y_pred)

'''