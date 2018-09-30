import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")
plt.boxplot(data["Day Calls"])
plt.ylabel("Numero de llamadas diarias")
plt.title("Boxplot de las llamadas diarias")
plt.show()