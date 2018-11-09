import numpy as np 
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans

data = np.random.random(90).reshape(30,3)
#print(data)

c1 = np.random.choice(range(len(data)))
c2 = np.random.choice(range(len(data)))
clust_centers = np.vstack([data[c1], data[c2]])
print(clust_centers)

vq(data,clust_centers)
print(vq)

#arroja el cluster correspondiente de cada elemento y la distancia al varicentro

