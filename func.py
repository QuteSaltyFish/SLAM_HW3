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
    
    def __str__(self):
        return str(self.data)

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

    def q2r(self):
        qw = self.data[0]
        qv = self.data[1:].view(-1,1)
        qv_hat = t.tensor([
            [0, -qv[2], qv[1]],
            [qv[2], 0, -qv[0]],
            [-qv[1], qv[0], 0]
        ], dtype=t.float, device=self.DEVICE)
        R = (qw**2 - t.matmul(qv.T, qv))*t.eye(3) + 2*t.matmul(qv, qv.T) + 2*qw*qv_hat
        return R

class data():
    def __init__(self,device=None):
        if device==None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.X_data = pd.read_table("src/rot_x.txt",sep=' ',header=None)
        self.X_data = t.tensor(self.X_data.values).to(self.DEVICE)

        self.Y_data = pd.read_table("src/rot_y.txt",sep=' ',header=None)
        self.Y_data = t.tensor(self.Y_data.values).to(self.DEVICE)
        

class problem1():
    def __init__(self, device=None):
        if device==None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = data(device=self.DEVICE)
        self.__calculate_X_and_Y()

    def __calculate_X_and_Y(self):
        new_X = []
        new_Y = []
        for row in range(self.data.X_data.shape[0]):
            x_tmp = t.tensor([
                [*self.data.X_data[row],0,0,0,0,0,0],
                [0,0,0, *self.data.X_data[row],0,0,0],
                [0,0,0,0,0,0,*self.data.X_data[row]]
            ]).to(self.DEVICE)
            y_tmp = t.tensor([
                *self.data.Y_data[row]
            ]).to(self.DEVICE)
            new_X.append(x_tmp)
            new_Y.append(y_tmp)
        self.new_X = t.cat(new_X, dim=0)
        self.new_Y = t.cat(new_Y, dim=0) 
    
    def sovle_R(self):
        new_X = self.new_X
        new_Y = self.new_Y
        r = t.matmul(t.matmul(t.inverse(t.matmul(new_X.T, new_X)), new_X.T), new_Y)
        r = r.view(3,3)
        print(r)
        # start using SVD to process the matrix
        u, s, v = t.svd(r)

        I = t.tensor([
            [1.0,0,0],
            [0,1,0],
            [0,0,1]
        ], dtype=t.float64).to(self.DEVICE)
        R = t.matmul(t.matmul(u,I), v)
        return R


class problem2():
    def __init__(self,device=None):
        if device==None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = data(device=self.DEVICE)
    
    
    def cal_A(self):
        A = t.zeros([4,4], dtype=t.float, device=self.DEVICE)
        for i in range(self.data.X_data.shape[0]):
            tmp_x  = quaternion([0,*self.data.X_data[i]],self.DEVICE)
            tmp_y  = quaternion([0,*self.data.Y_data[i]], self.DEVICE)
            A += t.matmul(tmp_y.left_Quaternion().T, tmp_x.right_Quaternion())
        return A
    
    def solve_q(self):
        A = self.cal_A()
        e, V = t.eig(A, True)
        print(e,V)
        idx = t.argmax(e[:,0])
        e1 = e[idx,0]
        q  = quaternion(V[idx], device=self.DEVICE)
        return e1,q
        
if __name__ == "__main__":
    
    time_start=time.time()
    x = quaternion([1,2,3,4])
    y = quaternion([5,6,7,8])
    x.q2r()
    # print(t.matmul(x.left_Quaternion(),y.data))
    # print(t.matmul(y.right_Quaternion(), x.data))
    time_end=time.time()
    print('time cost',time_end-time_start,'s')