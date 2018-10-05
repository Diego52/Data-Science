import numpy as np 
from numpy import linalg
import statsmodels.api as sm
'''Defiinir la funcion de entorno L(b)
L(b)= sum(Pi^Yi) * (1-Pi)^Yi     Pi = probabilidad de que ocurra Yi sabiendo que ocurrio Xi
'''

def linkelihood(y, pi):
    total_sum = 1
    sum_in = list(range(1, len(y)+1))
    for i in range(len(y)):
        sum_in[i] = np.where(y[i]==1, pi[i], 1-pi[i])
        total_sum = total_sum * sum_in[i]
    return total_sum

'''Calcular las probabilidades condicionales para cada observacion 
Pi = P(xi) = 1 / (1+ e^(-B*xi)) '''

def logitprobs(x,beta):
    n_rows = np.shape(x)[0]
    n_cols = np.shape(x)[1]
    pi = list(range ( 1, n_rows + 1))  #lo creamos como vector
    expo = list(range(1, n_rows +1))
    for i in range(n_rows):
        expo[i] = 0
        for j in range(n_cols):
            ex = x[i][j] * beta[j]
            expo[i] = ex + expo[i]
        with np.errstate(divide="ignore", invalid = "ignore"):
            pi[i] = 1/(1 + np.exp(-expo[i]))
    return pi

'''Calcular la matrix diagonal W
W = diag(Pi * (1-Pi)) '''

def findW(pi): 
    n = len(pi)
    W = np.zeros(n*n).reshape(n,n)
    for i in range (n):
        print(i)
        W[i,i] = pi[i] * (1-pi[i])
        W[i,i].astype(float)
    return W

'''Obtener la solucion de la funcion logistica

Bn+1 = Bn - F(B)/f(B)'''
def logistics(x,y,limit):
    nrow = np.shape(x)[0]
    bias = np.ones(nrow).reshape(nrow,1) #crea una matriz de puros 1 y la pasa a vector
    x_new = np.append(x,bias,axis=1)
    ncol = np.shape(x_new)[1]
    beta = np.zeros(ncol).reshape(ncol,1) #columna de 0
    root_dif = np.array(range(1,ncol+1)).reshape(ncol,1)
    iter_i = 10000
    while(iter_i >limit):
        print("Iter:i" + str(iter_i) + ", limit:" + str(limit))
        pi = logitprobs(x_new,beta)
        print("Pi" + str(pi))
        W = findW(pi)
        print("Str" + str(W))
        num = (np.transpose(np.matrix(x_new)) * np.matrix(y-np.transpose(pi)).transpose()) #los vecotres vienen en fila y se ocupa columna para multiplicar
        dem = (np.matrix(np.transpose(x_new))*np.matrix(W)*np.matrix(x_new))
        root_dif = np.array(linalg.inv(dem) * num)
        beta = beta + root_dif 
        print("Beta" + str(beta))
        iter_i = np.sum(root_dif*root_dif)
        ll = linkelihood(y, pi)
    return beta

'''Comprobacion experimental'''
x = np.array(range(10)).reshape(10,1)
y = [0,0,0,0,1,0,1,0,1,1] 

bias = np.ones(10).reshape(10,1)
x_new = np.append(x,bias,axis=1)
a = logistics(x, y, 0.00001)
ll = linkelihood(y, logitprobs(x,a))
print(ll)
#Y = 0.662208 * X -3.6955
'''Con el paquete statsmodel de python'''
logit_model = sm.Logit(y,x_new)
result = logit_model.fit()
print(result.summary())