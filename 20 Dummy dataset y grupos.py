import pandas as pd 
import numpy as np 

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
    }  #, index= range(40,40+n)
)
#print (data.describe())
print(data.head())

double_group = data.groupby(["Gender", "Economic Status"])
for names, groups in double_group:
    print(names)
    print(groups)
#print(double_group.groups)
print(double_group.get_group(('Female','Rich')))