import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

#importar el dataset
df = pd.read_csv('python-ml-course-master/datasets/wine/winequality-red.csv', sep = ';')
print(df.head())
plt.hist(df['quality'])
plt.show()
print(df.groupby('quality').mean())
#normaliar los datos
df_nom = (df-df.min())/(df.max()-df.min())
print(df_nom.head())
#cluster jerarquico con sckit learn
cluster = AgglomerativeClustering(n_clusters= 6 , linkage='ward').fit(df_nom)
md = pd.Series(cluster.labels_)
plt.hist(md)
plt.title('Histograma de los labels')
plt.xlabel('Cluster')
plt.ylabel('Numero de vinos del cluster')
plt.show()
print(cluster.children_)

Z = linkage(df_nom, "ward")
plt.figure(figsize=(25,10))
plt.title("Dendrograma de los vinos")
plt.xlabel("ID del vino")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=4.)
plt.show()

from sklearn.cluster import KMeans
from sklearn import datasets
model = KMeans(n_clusters=6)
model.fit(df_nom)
print(model.labels_)
md_k = pd.Series(model.labels_)
df_nom["clust_h"] = md_h
df_nom["clust_k"] = md_k
df_nom.head()
plt.hist(md_k)
print(model.cluster_centers_)
print(model.inertia_)
print(df_norm.groupby("clust_k").mean())