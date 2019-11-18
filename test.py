import numpy as np
from numpy import linalg as la

import numpy as np
from numpy import linalg as la
x_raw = np.loadtxt('src/pointdata/rot_x.txt')
y_raw = np.loadtxt('src/pointdata/rot_y.txt')
x_raw = x_raw - np.mean(x_raw,axis=0)
y_raw = y_raw - np.mean(y_raw,axis=0)
x_size = x_raw.shape[0]
X = np.zeros((3*x_size,9))
Y = np.zeros((3*x_size,1))
for i in range(0,x_size):
    for j in range(0,3):
        X[3*i,j] = x_raw[i,j]
        X[3*i+1,j+3] = x_raw[i,j]
        X[3*i+2,j+6] = x_raw[i,j]
        Y[3*i + j] = y_raw[i,j]
U,sigma,Vt = la.svd(X)
temp_y = np.matmul(U.T,Y)
temp_r = np.zeros((9,1))
#print(np.matmul(Vt,Vt.T))
for i in range(0,9):
    temp_r[i] = temp_y[i] / sigma[i]
r = np.matmul(Vt.T,temp_r)
r = r.reshape(3,3)
result = np.matmul(r,x_raw[10,:].reshape(3,1))
print(result,y_raw[10,:])
print(r)