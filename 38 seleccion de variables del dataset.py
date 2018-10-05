import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("python-ml-course-master/datasets/bank/bank.csv", sep=";")
data["y"] = (data["y"] == "yes").astype(int)
categories = ["job", "marital", "education","default", "housing", "loan", "contact", 
    "month", "day_of_week","poutcome"]
for category in categories: 
    cat_list = "cat" + "_" + category
    cat_dummies = pd.get_dummies(data[category], prefix=cat_list)
    data = data.join(cat_dummies)

data_vars = data.columns.values.tolist()
to_keep = [v for v in data_vars if v not in categories]
bank_data = data[to_keep]
bank_data_vars = bank_data.columns.values.tolist()
Y = ['y']
X = [v for v in bank_data_vars if v not in Y]
'''Seleccion de rasgos para el modelo'''
n = 12 
lr = LogisticRegression()
rfe = RFE(lr, n)
rfe = rfe.fit(bank_data[X],bank_data[Y].values.ravel())
print(rfe.support_)
print(rfe.ranking_)
z = zip(bank_data_vars,rfe.support_,rfe.ranking_)
print(list(z))
cols = ["previous", "euribor3m", "cat_job_blue-collar", "cat_job_retired", "cat_month_aug", "cat_month_dec", "cat_month_jul", 
"cat_month_jun", "cat_month_mar", "cat_month_nov", "cat_day_of_week_wed", "cat_poutcome_nonexistent"]

X = bank_data[cols]  
Y = bank_data["y"]

