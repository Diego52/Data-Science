import numpy as np 
import matplotlib.pyplot as plt

data = np.random.randn(1000000)
x = range(1,1000001)
plt.plot(x,data)
plt.show()
plt.hist(data,bins=int(1 + np.log2(3333)) )
plt.show()
plt.plot(x,sorted(data))
plt.show()
mu = 5.5
sd = 2.5
z_10000 = np.random.randn(10000)
data = mu + sd*z_10000
plt.hist(data,bins=int(1 + np.log2(3333)) )
plt.show()
data = np.random.randn(2,4)
