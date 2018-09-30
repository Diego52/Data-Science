import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn

'''Manera RANDOM'''
data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")

a = np.random.randn(len(data))
plt.hist(a)
#plt.show()

check = (a<0.8)
plt.hist(check)
#plt.show()

training = data[check]
testing = data[~check]

'''CON SKLEARN'''

train, test = train_test_split(data, test_size = 0.2)
#print(len(train),len(test))

'''USANDO FUNCION DE SHUFFLE'''

data = sklearn.utils.shuffle(data)
cut_id = int(0.8 * len(data))
train_data = data[:cut_id]
test_data = data[cut_id+1:]
#print(len(train_data),len(test_data))

