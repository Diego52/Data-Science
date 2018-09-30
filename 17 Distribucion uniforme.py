import numpy as np 
import matplotlib.pyplot as plt
a = 1
b = 100
n = 1000000 
data = np.random.uniform(a,b,n)
plt.hist(data, bins=int(1 + np.log2(3333)))
plt.show()