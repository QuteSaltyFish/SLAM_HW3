import torch as t 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd 
import time
from mpl_toolkits.mplot3d import Axes3D 
import time


class quaternion():
    def __init__(self, data, device=None):
        """
        The input data is a numpy data, or list
        """
        if device==None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = t.tensor(data, dtype=t.float).to(self.DEVICE)
    
    def left_Quaternion(self):
        x = self.data
        out = t.tensor([
            [x[0], -x[1], -x[2], -x[3]],
            [x[1], x[0], -x[3], x[2]],
            [x[2], x[3], x[0], -x[1]],
            [x[3], -x[2], x[1], x[0]]
        ],dtype=t.float, device=self.DEVICE)
        return out

    def right_Quaternion(self):
        x = self.data
        out = t.tensor([
            [x[0], -x[1], -x[2], -x[3]],
            [x[1], x[0], x[3], -x[2]],
            [x[2], -x[3], x[0], x[1]],
            [x[3], x[2], -x[1], x[0]]
        ],dtype=t.float, device=self.DEVICE)
        return out

class data():
    def __init__(self,device=None):
        if device==None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.X_data = pd.read_table("src/rot_x.txt",sep=' ',header=None)
        self.X_data = t.tensor(self.X_data.values).to(self.DEVICE)

        self.Y_data = pd.read_table("src/rot_x.txt",sep=' ',header=None)
        self.Y_data = t.tensor(self.Y_data.values).to(self.DEVICE)
        # %%
        new_X = []
        new_Y = []
        for row in range(self.X_data.shape[0]):
            x_tmp = t.tensor([
                [*self.X_data[row],0,0,0,0,0,0],
                [0,0,0, *self.X_data[row],0,0,0],
                [0,0,0,0,0,0,*self.X_data[row]]
            ]).to(self.DEVICE)
            y_tmp = t.tensor([
                *self.Y_data[row]
            ]).to(self.DEVICE)
            new_X.append(x_tmp)
            new_Y.append(y_tmp)
        self.new_X = t.cat(new_X, dim=0)
        self.new_Y = t.cat(new_Y, dim=0) 

class problem1():
    def __init__(self, device=None):
        if device==None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = data(device=self.DEVICE)
        
    def sovle_R(self):
        new_X = self.data.new_X
        new_Y = self.data.new_Y
        r = t.matmul(t.matmul(t.inverse(t.matmul(new_X.T, new_X)), new_X.T), new_Y)
        r = r.view(3,3)
        
        # start using SVD to process the matrix
        u, s, v = t.svd(r)

        I = t.tensor([
            [1.0,0,0],
            [0,1,0],
            [0,0,1]
        ], dtype=t.float64).to(self.DEVICE)
        R = t.matmul(t.matmul(u,I), v)
        return R
if __name__ == "__main__":
    
    time_start=time.time()
    x = quaternion([1,2,3,4])
    y = quaternion([5,6,7,8])
    print(t.matmul(x.left_Quaternion(),y.data))
    print(t.matmul(y.right_Quaternion(), x.data))
    time_end=time.time()
    print('time cost',time_end-time_start,'s')