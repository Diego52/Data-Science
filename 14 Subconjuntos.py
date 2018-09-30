import pandas as pd


data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")

'''Crear un subconjunto de datos'''

account_length = data['Account Length']
#print(account_length.head())
#print(type(account_length))

subset = data[['Account Length', 'Phone', 'Day Calls']]
#print(subset.head())
#print(type(subset))

desired_columns = ['Account Length', 'Phone', 'Day Calls']
all_columns_list = data.columns.values.tolist()
sublist = [x for x in all_columns_list if x not in desired_columns]
'''
   a = set(desired_columns)
    b = set(all_columns_list)
    sublist = b-a
    sublist = list(sublist)
'''
#print(data[sublist])
#print(data[0:10]) #data[:10]   data[3320:]

'''Condicionales como filtro de datos'''

data1 = data[data["Day Mins"]>330]
print(data1)

data2 = data[data["State"] == "NY"]
print(data2)