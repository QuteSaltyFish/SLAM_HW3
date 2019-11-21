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
import func

os.environ["CUDA_VISIBLE_DEVICES"] ='0'

solution = func.problem3("cpu",'1')

# solve the problem using the rotation matrix
print('-'*100)

solution.Gauss_Newton_R(100)
# solve the problem using the quaterion
print('-'*100)
solution.Gauss_Newton_Q(100)

