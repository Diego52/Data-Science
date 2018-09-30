import pandas as pd


data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")
subset_first_50 = data[["Day Mins","Night Mins","Account Length"]][:50]
#print(subset_first_50)
print(data.iloc[:10,3:6]) #[:,3:6] todas las filas de la columna 3 a la 6
print(data.loc[:10, ["Phone","Area Code"]])
data["Total Calls"] = data["Day Calls"] + data["Night Calls"] + data["Eve Calls"]
#print(data["Total Calls"].head())