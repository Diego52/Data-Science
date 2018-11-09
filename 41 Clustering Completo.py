'''
X dataset (array de nxm) de puntos a clusterizar
n numero de datos
m numero de rasgos
z array de enlace del cluster con la info de las uniones
k numero de clusters '''

import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import inconsistent
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

'''Exploracion de datos'''

np.random.seed(4711)
a = np.random.multivariate_normal([10,0],[[3,1] ,[1,4]], size = [100,])
#normal con componente en x una normal [10,0], como componente siguiente [[3,1],[1,4]]
b =  np.random.multivariate_normal([0,20],[[3,1] ,[1,4]], size = [50,])

x = np.concatenate((a,b))
print(x.shape)
plt.scatter(x[:,0], x[:,1])
plt.show()

z = linkage(x, "ward")

c, coph_dist = cophenet(z, pdist(x)) #coeficiente de correlacion de las nuevas distancias con las originales

print(z[152-len(x)])
print(z[158-len(x)])

print(x [[33,62, 68]])
idx = [33,62,68]
idx2 = [15,69,41]
plt.figure(figsize =(10,8))
plt.scatter(x[:,0],x[:,1])
plt.scatter(x[idx,0], x[idx,1], c="r")
plt.scatter(x[idx2,0], x[idx2,1], c="y")
plt.show()

'''Representacion de un dendrograma'''

plt.figure(figsize = (25,10))
plt.title("Dendrograma del clustering jerarquico")
plt.xlabel("Indices de la muestra")
plt.ylabel("Distancias")
dendrogram(z, leaf_rotation=90.,leaf_font_size=8.0) #color_threshold=0.1*180
plt.show()

#Truncar el dendrograma

plt.figure(figsize = (25,10))
plt.title("Dendrograma del clustering jerarquico truncado")
plt.xlabel("Indices de la muestra")
plt.ylabel("Distancias")
dendrogram(z, leaf_rotation=90.,leaf_font_size=12.0, truncate_mode="lastp",p=12, show_leaf_counts=True,show_contracted=True) #color_threshold=0.1*180
plt.show()

#Dendrograma tuneado
class Dendogram:
    def dendrogram_tune(*args, **kwargs):
        max_d = kwargs.pop("max_d", None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
        ddata = dendrogram(*args, **kwargs)
        if not kwargs.get('no_plot', False):
            plt.title('Clustering jerarquico con Dendrograma truncado')
            plt.xlabel('Indice del dataset (o tamaño del cluster)')
            plt.ylabel('Distancia')
            for index, distance, color in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5*sum(index[1:3])
                y = distance[1]
                if y > annotate_above:
                    plt.plot(x,y,'o', c=color)
                    plt.annotate('%.3g'%y,(x,y), xytext = (0, -5),
                        textcoords="offset points", va="top", ha="center")
        if max_d:
            plt.axhline(y=max_d, c='k')
        return 

dendrogram_tune(z,truncate_mode = 'lastp', p=12, leaf_rotation=90., 
    leaf_font_size=12., show_contracted = True, annotate_above = 10, max_d = 20)
plt.show()

'''Metodo de corte automatico del dendrograma
inconsistency:i = (h_i-avg(h_i))/std(h_i)
donde i es cada union de clusters (linea horizontal)
'''
depth = 5 #cantidad de clusters abajo para realizar la inconsistencia
incons = inconsistent(z, depth)
incons[-10:]
#no funciona por que no hay una distribucion normal entonces depende mucho de la profundidad especificada
#muy probablemente la union a analizar se considere un outlier con respecto a las anteriores
#cada union toma mas distancia en realizarse

'''El metodo del codo'''

last = z[-10:,2]
last_rev = last[::-1] #los revertimos, se toman todos y el -1 indica que el ultimo pasa a ser el primero
acc = np.diff(last,2) #de dos en dos, osea el primero con el segundo, el segundo con el tercero, este con el cuarto y asi
acc_rev = acc[::-1]
idx = np.arange(1, len(last)+1)
plt.plot(idx, last_rev)
plt.plot(idx[:-2]+1, acc_rev)
plt.show()
k = acc_rev.argmax() + 2
print('El numero optimo de cluster es %s' %str(k))
c = np.random.multivariate_normal([40,40], [[20,1], [1,30]], size = [200,])
d = np.random.multivariate_normal([80,80], [[30,1], [1,30]], size = [200,])
e = np.random.multivariate_normal([0,100], [[100,1], [1,100]], size = [200,])
x2 = np.concatenate((x,c,d,e,))
plt.scatter(x2[:,0],x2[:,1])
plt.show()
z2 = linkage(x2,'ward')
plt.figure(figsize=(10,10))
dendrogram_tune(
    z2,
    truncate_mode = 'lastp',
    p = 30,
    leaf_rotation = 90.,
    leaf_font_size = 10.,
    show_contracted = True,
    annotate_above = 40,
    max_d = 170
)
plt.show()
last = z2[-10:,2]
last_rev = last[::-1] #los revertimos, se toman todos y el -1 indica que el ultimo pasa a ser el primero
acc = np.diff(last,2) #de dos en dos, osea el primero con el segundo, el segundo con el tercero, este con el cuarto y asi
acc_rev = acc[::-1]
plt.plot(idx[:-2]+1, acc_rev)
idx = np.arange(1, len(last)+1)
plt.plot(idx, last_rev)
plt.show()
k = acc_rev.argmax() + 2
print('El numero optimo de cluster es %s' %str(k))
print(inconsistent(z2,5)[-10:])
#consultar el notebook
'''Recuperar los clusters'''
max_d = 20
clusters = fcluster(z, max_d, criterion='distance')
print(clusters)

k = 3
clusters = fcluster(z, k, criterion='maxclust')
print(clusters)

print(fcluster(z,8,depth=10))

plt.figure(figsize=(10,8))
plt.scatter(x[:,0], x[:,1], c = clusters, cmap = 'prism')
plt.show()

max_d = 170
clusters = fcluster(z2, max_d, criterion='distance')
plt.figure(figsize=(10,8))
plt.scatter(x2[:,0], x2[:,1], c = clusters, cmap = 'prism')
plt.show()
