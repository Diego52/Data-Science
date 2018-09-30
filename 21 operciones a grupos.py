import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
gender = ["Male","Female"]
income = ["Poor", "Middle Class", "Rich"]
gender_data = []
income_data = []
n = 500 

for i in range (0, n): 
    gender_data.append(np.random.choice(gender))
    income_data.append(np.random.choice(income))

height = 160 + 30 * np.random.randn(n)
weight = 65 + 25 * np.random.randn(n)
age = 30 + 12 * np.random.randn(n)
income = 18000 + 3500 * np.random.randn(n)

for j in range (0,len(age)):
    age[j] = int(age[j])
    income[j] = int(income[j])


data = pd.DataFrame(
    {
        'Gender' : gender_data,
        'Economic Status' : income_data,
        'Height' : height,
        'Weight' : weight,
        'Age' : age,
        'Income': income 
    }
)
double_group = data.groupby(["Gender", "Economic Status"])

double_group.sum()
double_group.mean()
double_group.size()
double_group.describe()
double_group["Income"].sum()

x = double_group.aggregate(
    {
        "Income": np.sum,
        "Age": np.mean,
        "Height": lambda h: (np.mean(h) )/ np.std(h)
    }
)
#x = double_group.aggregate([np.std,np.sum,np.mean])
#x = double_group.aggregate([lambda x: np.mean(x) / np.std(x)])
'''Filtrado'''
#print(double_group['Age'].filter(lambda x: x.sum()>2400))
'''Tranformacion de variables'''
zscore = lambda x : (x - x.mean())/x.std()
z_group = double_group.transform(zscore)
plt.hist(z_group['Age'])
plt.show()

'''
fill_na_mean = lambda x : x.fillna(x.mean())
double_group.transform(fill_na_mean)
'''
#double_group.head(1) devuelve la primer fila de cada grupo, tail para el ultimo
#double_group.nth(32) regresa el elemento seleccionado *no es df

'''Ordenado'''

data_sorted = data.sort_values(['Age', 'Income'])
data_sorted.head(10)
age_grouped = data_sorted.groupby('Gender')