import torch as t 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd 
import time
from mpl_toolkits.mplot3d import Axes3D 
import time

os.environ["CUDA_VISIBLE_DEVICES"] ='0'


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
        
if __name__ == "__main__":
    time_start=time.time()
    x = quaternion([1,2,3,4],device="cpu")
    y = quaternion([5,6,7,8],device="cpu")
    print(t.matmul(x.left_Quaternion(),y.data))
    print(t.matmul(y.right_Quaternion(), x.data))
    time_end=time.time()
    print('time cost',time_end-time_start,'s')