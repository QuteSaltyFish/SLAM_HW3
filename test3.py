import numpy as np
import math

def txtSolver(path_x, path_y):
    r'''
    读取txt文件中的点云位置
    '''
    f_x = open(path_x)
    f_y = open(path_y)

    x_lines = f_x.readlines()
    y_lines = f_y.readlines()
    
    # 获取坐标并且减去平均值
    cor_X = np.zeros([len(x_lines), 3], np.float32)
    cor_Y = np.zeros([len(y_lines), 3], np.float32)

    for (i, line) in enumerate(x_lines):
        cor_X[i, :] = line.split()
    cor_X = cor_X - np.mean(cor_X, axis=0)
    
    for (i, line) in enumerate(y_lines):
        cor_Y[i, :] = line.split()
    cor_Y = cor_Y - np.mean(cor_Y, axis=0)

    # 获取真正所需的X、Y矩阵
    X = cor_X
    Y = cor_Y

    return X, Y

def get_X(vector):
    [x1, x2, x3] = vector[:]
    X = [
        [0, -x3, x2],
        [x3, 0, -x1],
        [-x2, x1, 0]
    ]

    return np.asarray(X)

'''主函数'''
if __name__ == '__main__':
    path_x = 'src/pointdata/zxl_x.txt'
    path_y = 'src/pointdata/zxl_y.txt'
    X, Y = txtSolver(path_x, path_y)

    # 这里设置初始的R
    '''
    R = np.array([
    [-9.97730512e-01, -6.73336586e-02, 5.96927957e-05],
    [ 6.73336834e-02, -9.97730292e-01, 6.63615484e-04],
    [ 1.48736520e-05, 6.66128753e-04, 9.99999778e-01]]
    ).astype(np.float32)
    '''
    R = np.identity(3)

    # pre_R
    pre_R = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
    ]).astype(np.float32)

    delta_theta = np.zeros([3, 1])
    #pre_delta_theta = np.ones_like(delta_theta)

    J = np.zeros([3*X.shape[0], 3])

    z = np.zeros([3*X.shape[0], 1])
    j = 0

    while(True):
        j += 1
        print('iter ' + str(j) + ' :')
        for i in range(X.shape[0]):
            J[i*3:i*3+3] = -1 * get_X(np.matmul(R, np.reshape(X[i], [3,1])))
            z[i*3:i*3+3] = np.reshape(Y[i], [3,1]) - np.matmul(R, np.reshape(X[i], [3,1]))
    
        delta_theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(J.T, J)), J.T),z)
        exp = np.identity(3) + get_X(delta_theta)
        R = np.matmul(R, exp)
        
        # correction with svd
        U, sigma, V = np.linalg.svd(R)
        sigma = np.identity(3)
        R = np.matmul(np.matmul