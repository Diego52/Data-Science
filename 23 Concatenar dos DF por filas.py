import pandas as pd 

red_wine = pd.read_csv("python-ml-course-master/datasets/wine/winequality-red.csv", sep=";")

white_wine = pd.read_csv("python-ml-course-master/datasets/wine/winequality-white.csv", sep=";")

''' En python tenemos dos ejes:
axis 0 denota horizontal
axis 1 denota vertical
'''
wine_data = pd.concat([red_wine,white_wine], axis= 0)

#print(wine_data.shape)

'''IMPORTAR CIENTOS DE FICHEROS JUNTOS'''
filepath = "python-ml-course-master/datasets/distributed-data/"
data = pd.read_csv(filepath + '001.csv')
for i in range (2,333):
    if i < 10:
        filename = '00' + str(i)
    elif i < 100:
        filename = '0' + str(i)
    else:
        filename = str(i)
    files = filepath + filename + '.csv'
    temp_data = pd.read_csv(files)
    data = pd.concat([data, temp_data], axis = 0)
print(data.shape)