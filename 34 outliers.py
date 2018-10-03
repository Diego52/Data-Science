import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

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
lm = LinearRegression()
data["displacement"] = data["displacement"].fillna(data["displacement"].mean())
data["mpg"] = data["mpg"].fillna(data["mpg"].mean())
plt.plot(data["displacement"], data["mpg"], "ro")
plt.show()
lm.fit(data[["displacement"]],  data[["mpg"]])
y_pred = lm.predict(data[["displacement"]])

model_validation(data[["displacement"]],data[["mpg"]],y_pred)

outliers = data[((data["displacement"]>250)&(data["mpg"]>35)) | ((data["displacement"]>330)&(data["mpg"]>20))]
print(outliers) 

data = data.drop([11,12,13,14,305,372,395])
lm = LinearRegression()
data["displacement"] = data["displacement"].fillna(data["displacement"].mean())
data["mpg"] = data["mpg"].fillna(data["mpg"].mean())
plt.plot(data["displacement"], data["mpg"], "ro")
plt.show()
lm.fit(data[["displacement"]],  data[["mpg"]])
y_pred = lm.predict(data[["displacement"]])

model_validation(data[["displacement"]],data[["mpg"]],y_pred)

