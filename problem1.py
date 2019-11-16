#%%
import torch as t 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd 
import time
from mpl_toolkits.mplot3d import Axes3D 
import time

time_start=time.time()


os.environ["CUDA_VISIBLE_DEVICES"] ='1'
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
DEVICE = t.device("cpu")
#%%
X_data = pd.read_table("src/rot_x.txt",sep=' ',header=None)
X_data = t.tensor(X_data.values).to(DEVICE)

Y_data = pd.read_table("src/rot_x.txt",sep=' ',header=None)
Y_data = t.tensor(Y_data.values).to(DEVICE)


# %%
new_X = []
new_Y = []
for row in range(X_data.shape[0]):
    x_tmp = t.tensor([
        [*X_data[row],0,0,0,0,0,0],
        [0,0,0, *X_data[row],0,0,0],
        [0,0,0,0,0,0,*X_data[row]]
    ]).to(DEVICE)
    y_tmp = t.tensor([
        *Y_data[row]
    ]).to(DEVICE)
    new_X.append(x_tmp)
    new_Y.append(y_tmp)
new_X = t.cat(new_X, dim=0)
new_Y = t.cat(new_Y, dim=0) 
# %%
r = t.matmul(t.matmul(t.inverse(t.matmul(new_X.T, new_X)), new_X.T), new_Y)
r = r.view(3,3)
# %%
# start using SVD to process the matrix
u, s, v = t.svd(r)

# %%
I = t.tensor([
    [1.0,0,0],
    [0,1,0],
    [0,0,1]
], dtype=t.float64).to(DEVICE)
R = t.matmul(t.matmul(u,I), v)
# %%
print(R)
time_end=time.time()
print('time cost',time_end-time_start,'s')
# %%
