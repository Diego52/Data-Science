import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
data_Ads = pd.read_csv('python-ml-course-master/datasets/ads/Advertising.csv')
print(data_Ads.corr())
plt.matshow(data_Ads.corr())
plt.show()
def corr_coeff(df,var1,var2):

    df['corrn'] = (df[var1] - np.mean(df[var1])) * (df[var2]- np.mean(df[var2]))
    df['corr1'] = ((df[var1]) - np.mean(df[var1]))**2
    df['corr2'] = ((df[var2]) - np.mean(df[var2]))**2
    corr_pearson = sum(df['corrn'])/np.sqrt(sum(df['corr1']) * sum(df['corr2']))
    return corr_pearson
cols = data_Ads.columns.values
for x in cols:
    for y in cols:
        print(x + ',' + y + ':' + str(corr_coeff(data_Ads,x,y)))
plt.plot(data_Ads['TV'],data_Ads['Sales'], "ro")
plt.title('Gasto en TV vs Ventas del producto')
plt.show()
