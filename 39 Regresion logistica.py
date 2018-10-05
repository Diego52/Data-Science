import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
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
cols = ["previous", "euribor3m", "cat_job_blue-collar", "cat_job_retired", "cat_month_aug", "cat_month_dec", "cat_month_jul", 
"cat_month_jun", "cat_month_mar", "cat_month_nov", "cat_day_of_week_wed", "cat_poutcome_nonexistent"]
X = bank_data[cols]  
Y = bank_data["y"]  

logit_model = sm.Logit(Y,X)
result = logit_model.fit()
print(result.summary())

#Implementacion del modelo con scikit learn

logit_model = LogisticRegression()
logit_model.fit(X,Y)
print(logit_model.score(X,Y))

'''
Descubri lo que hace .score :O
data["ye"] = logit_model.predict(X)
data["result"] = data["ye"] == data["y"]
a = data["result"].astype(int).sum()
b = data.shape[0]
print(a/b)
'''
#Si p<0.5 -> 0
#Si p>0.5 -> 1
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.3, random_state = 0)
rl = LogisticRegression()
rl.fit(X_train, Y_train)
probs = rl.predict_proba(X_test)
prediccion = rl.predict(X_test)
prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df["prediction"] = np.where(prob_df[0]>threshold,1,0)
print(prob_df.head())
print(pd.crosstab(prob_df.prediction, columns="count"))

from sklearn import metrics
print ("Acertamos en el",metrics.accuracy_score(Y_test, prediccion) * 100,"porciento de los casos")