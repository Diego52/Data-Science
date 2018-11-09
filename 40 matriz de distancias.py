from  scipy.spatial import distance_matrix
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv("python-ml-course-master/datasets/movies/movies.csv", sep=";")
print(data)
movies = data.columns.values.tolist()[1:]
print(movies)
dd1 = distance_matrix(data[movies],data[movies],p=1)
dd2 = distance_matrix(data[movies],data[movies],p=2)
dd10 = distance_matrix(data[movies],data[movies],p=10)

def dm_to_df(dd,col_name):
    return pd.DataFrame(dd,index=col_name, columns =col_name)

print(dm_to_df(dd1, data["user_id"]))
print(dm_to_df(dd2, data["user_id"]))
print(dm_to_df(dd10, data["user_id"]))
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(xs = data["star_wars"],ys = data["lord_of_the_rings"], zs = data["harry_potter"])
plt.show()

''' Clustering jerarquico a mano '''
df = dm_to_df(dd1,data["user_id"])

z=[]

df[11] = df[1] + df[10]
df.loc[11] = df.loc[1] + df.loc[10]
z.append([1,10,0.7,2]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[11][i] = min(df.loc[1][i], df.loc[10][i])
    df.loc[i][11] = min(df.loc[i][1], df.loc[i][10])

df = df.drop([1,10])
df = df.drop([1,10], axis = 1)

x= 2
y= 7
n = 12
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
z.append([x,y,df.loc[x][y],2]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis = 1)

x= 5
y= 8
n = 13
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
z.append([x,y,df.loc[x][y],2]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis = 1)

x= 11
y= 13
n = 14
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
z.append([x,y,df.loc[x][y],2]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis = 1)

x= 9
y= 12
w = 14
n = 15
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
z.append([x,y,df.loc[x][y],3]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[w][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[i][w])

df = df.drop([x,y,w])
df = df.drop([x,y,w], axis = 1)

x= 4
y= 6
w = 15
n = 16
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
z.append([x,y,df.loc[x][y],3]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[w][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[i][w])

df = df.drop([x,y,w])
df = df.drop([x,y,w], axis = 1)

x= 3
y= 16
n = 17
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
z.append([x,y,df.loc[x][y],2]) #id1, id2, d, n_elementos -> 11

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis = 1)
print(df)

'''Clustering jerarquico'''

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

z = linkage(data[movies], "ward") #average,complete,single
plt.figure(figsize = (25,10))
plt.title("Dendrograma jerarquico para el clustering")
plt.xlabel("ID de los usuarios")
plt.ylabel("Distancia")
dendrogram(z, leaf_rotation = 90.,leaf_font_size=10)
plt.show()