import pandas as pd
import numpy as np
import random

data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")
#print(np.random.randint(1,100)) #np.random.random() numero entre 0-1

def rand_list(n,a,b):
    x = []
    for i in range (n):
        x.append(np.random.randint(a,b))
    return x

#print(random.randrange(0,100,7))

#SHUFFLING
a = np.arange(100)
#print(a)
np.random.shuffle(a)
#print(a)
columns = data.columns.values.tolist()
print(columns)
print(np.random.choice(columns))

##SEED
np.random.seed(2018)
for i in range(5):
    print(np.random.random())