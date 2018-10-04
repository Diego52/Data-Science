import pandas as pd 

df = pd.read_csv("python-ml-course-master/datasets/gender-purchase/Gender Purchase.csv")
print(df.head())

'''Tabla de contigencia
Conjunto de filas y columnas que dicen con que frecuencia ocurre un determinado suceso en la diferente combinacion
de categorias para dos o mas variables
'''

contingency_table = pd.crosstab(df["Gender"],df["Purchase"])

print(contingency_table)
print(contingency_table.sum(axis =1))
print(contingency_table.sum(axis =0))
print(contingency_table.astype("float").div(contingency_table.sum(axis =1), axis = 0)) #divide cada elemento entre la suma de esa fila, calculando asi su probabilidad

'''Probabilidad condicional (4 aqui)
Define si un suceso sea verdad o no de acuerdo a cosas que ya estan pasando

Cual es la probabilidad de que un cliente compre un producto sabiendo que es un hombre?
Cual es la probabilidad de que un sabiendo que un cliente compra un producto sea mujer?

P(Purchase|Male) = Numero total de compras hechas por hombres / Numero total de hombres del grupo
= Purchase n Male / Male
121/246
0.4918

P(Female|Purchase) = Numero total de compras hechas por mujeres / Numero total de compras
= Female n Purchase /Purchase
159/280
0.5678
'''

'''Ratio de probabilidades (2 aqui)
El cociente entre los casos de exito sobre los de fracaso en el suceso estudiado y para cada grupo
Pm = probabilidad de hacer una compra sabiendo que es hombre
P_f = probabilidad de hacer una compra sabiendo que es mujer
odds_purchase,male = Pm / 1-Pm = N_p,m /N_~p,m
odds_purchase,female = Pf / 1-Pf
'''
pm = 121/246
pf = 159/265
odd_m = pm/(1-pm)  #121/125
odd_f = pf/(1-pf) #159/106

#Si el ratio es superior a 1 es mas probable el exito
#Si el ratio es menor a 1 es mas probable el fracaso
#Si el ratio es igual a 1 es mas equiprobable

odd_r = odd_m/odd_f