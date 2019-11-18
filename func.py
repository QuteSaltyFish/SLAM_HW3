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
        if device == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = t.tensor(data, dtype=t.float, device=self.DEVICE)

    def __str__(self):
        return str(self.data)

    def left_Quaternion(self):
        x = self.data
        out = t.tensor([
            [x[0], -x[1], -x[2], -x[3]],
            [x[1], x[0], -x[3], x[2]],
            [x[2], x[3], x[0], -x[1]],
            [x[3], -x[2], x[1], x[0]]
        ], dtype=t.float, device=self.DEVICE)
        return out

    def right_Quaternion(self):
        x = self.data
        out = t.tensor([
            [x[0], -x[1], -x[2], -x[3]],
            [x[1], x[0], x[3], -x[2]],
            [x[2], -x[3], x[0], x[1]],
            [x[3], x[2], -x[1], x[0]]
        ], dtype=t.float, device=self.DEVICE)
        return out

    def q2r(self):
        qw = self.data[0]
        qv = self.data[1:].view(-1, 1)
        qv_hat = t.tensor([
            [0, -qv[2], qv[1]],
            [qv[2], 0, -qv[0]],
            [-qv[1], qv[0], 0]
        ], dtype=t.float, device=self.DEVICE)
        R = (qw**2 - t.matmul(qv.T, qv))*t.eye(3) + \
            2*t.matmul(qv, qv.T) + 2*qw*qv_hat
        return R


class data():
    def __init__(self, device=None, dataset='0'):
        if device == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        if dataset == '0':
            x_dir = "src/pointdata/rot_x.txt"
            y_dir = "src/pointdata/rot_y.txt"
        elif dataset == '1':
            x_dir = "src/pointdata/zxl_x.txt"
            y_dir = "src/pointdata/zxl_y.txt"

        self.X_data = pd.read_table(x_dir, sep=' ', header=None)
        self.X_data = t.tensor(self.X_data.values).to(self.DEVICE)
        self.X_data -= t.mean(self.X_data, dim=0)

        self.Y_data = pd.read_table(y_dir, sep=' ', header=None)
        self.Y_data = t.tensor(self.Y_data.values).to(self.DEVICE)
        self.Y_data -= t.mean(self.Y_data, dim=0)


class problem1():
    def __init__(self, device=None, dataset='0'):
        if device == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = data(device=self.DEVICE, dataset=dataset)
        self.__calculate_X_and_Y()

    def __calculate_X_and_Y(self):
        new_X = []
        new_Y = []
        for row in range(self.data.X_data.shape[0]):
            x_tmp = t.tensor([
                [*self.data.X_data[row], 0, 0, 0, 0, 0, 0],
                [0, 0, 0, *self.data.X_data[row], 0, 0, 0],
                [0, 0, 0, 0, 0, 0, *self.data.X_data[row]]
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
        r = t.matmul(
            t.matmul(t.inverse(t.matmul(new_X.T, new_X)), new_X.T), new_Y)
        r = r.view(3, 3)
        # start using SVD to process the matrix
        u, s, v = t.svd(r)
        R = t.matmul(t.matmul(u, t.eye(3, dtype=t.double)), v.T)
        return R

    def validate(self):
        R = self.sovle_R()
        new_Y = t.matmul(R, self.data.X_data.T)
        print('The difference is: {}'.format(
            t.dist(self.data.Y_data.T, new_Y)))


class problem2():
    def __init__(self, device=None, dataset='0'):
        if device == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = data(device=self.DEVICE, dataset=dataset)

    def cal_A(self):
        A = t.zeros([4, 4], dtype=t.float, device=self.DEVICE)
        for i in range(self.data.X_data.shape[0]):
            tmp_x = quaternion([0, *self.data.X_data[i]], self.DEVICE)
            tmp_y = quaternion([0, *self.data.Y_data[i]], self.DEVICE)
            A += t.matmul(tmp_y.left_Quaternion().T, tmp_x.right_Quaternion())
        return A

    def solve_q(self):
        A = self.cal_A()
        print('A Matrix:')
        print(A)
        e, V = t.eig(A, True)
        idx = t.argmax(e[:, 0])
        e1 = e[idx, 0]
        q = quaternion(V[:, idx], device=self.DEVICE)
        return e1, q

    def validate(self):
        R = self.solve_q()[1].q2r()
        new_Y = t.matmul(R, self.data.X_data.T.float())
        print('The difference is: {}'.format(
            t.dist(self.data.Y_data.T, new_Y.double())))

class problem3():
    def __init__(self, device=None, dataset='0'):
        if device == None:
            self.DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            self.DEVICE = t.device(device)

        self.data = data(device=self.DEVICE, dataset=dataset)

    def x2hat(self, x):
        """
        The input array is supposed to be of shape [3,1]
        """
        x_hat = t.tensor([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ], device=self.DEVICE)
        return x_hat
    
    def Gauss_Newton(self, epoch=10):
        R = t.ones([3,3], dtype=t.double, device=self.DEVICE)
        u, s, v = t.svd(R)
        R = t.matmul(t.matmul(u, t.eye(3, dtype=t.double, device=self.DEVICE)), v.T)

        for _ in range(epoch):
            dTheta = self.dTheta(self.data.X_data, self.data.Y_data, R)
            # tmp = t.exp()
            # R = R*tmp

            exp = t.eye(3, dtype=t.double, device=self.DEVICE) + self.x2hat(dTheta)
            R = np.matmul(R, exp)
            
            u, s, v = t.svd(R)
            R = t.matmul(t.matmul(u, t.eye(3, dtype=t.double, device=self.DEVICE)), v.T)
            self.validate(R)

    def dTheta(self, X, Y, R):
        J = []
        Z = []
        for i in range(X.shape[0]):
            dJ = -self.x2hat(t.matmul(R, X[i].T))
            dZ = Y[i] - t.matmul(R, X[i].T)
            J.append(dJ)
            Z.append(dZ)
        J = t.cat(J, dim=0)
        Z = t.cat(Z, dim=0)

        dTheta = t.matmul(t.matmul(t.inverse(t.matmul(J.T, J)), J.T), Z)
        return dTheta.view(-1,1)

    def validate(self, R=None):
        if type(R)== None:
            R = self.solve_q()[1].q2r()
        new_Y = t.matmul(R, self.data.X_data.T)
        print('The difference is: {}'.format(
            t.dist(self.data.Y_data.T, new_Y.double())))

if __name__ == "__main__":

    time_start = time.time()
    x = quaternion([1, 2, 3, 4])
    y = quaternion([5, 6, 7, 8])
    print(t.matmul(x.left_Quaternion(), x.left_Quaternion().T))
    print(t.matmul(y.left_Quaternion(), y.left_Quaternion().T))
    # print(t.matmul(x.left_Quaternion(),y.data))
    # print(t.matmul(y.right_Quaternion(), x.data))
    time_end = time.time()
    print('time cost', time_end-time_start, 's')
