import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("python-ml-course-master/datasets/customer-churn-model/Customer Churn Model.txt")
figure, axs = plt.subplots(2,2,sharex=True,sharey=True)
data.plot(kind="scatter", x="Day Mins", y="Day Charge", ax=axs[0][0])
data.plot(kind="scatter", x="Night Mins", y="Night Charge", ax=axs[0][1])
data.plot(kind="scatter", x="Day Calls", y="Day Charge", ax=axs[1][0])
data.plot(kind="scatter", x="Night Calls", y="Night Charge", ax=axs[1][1])
plt.show()


