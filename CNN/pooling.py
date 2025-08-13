'''
pooling无非就是降维
1. max pooling: 返回滑动窗口每一次滑动的最大值

'''
import torch
from torch import nn


# pooling的实现

def pool2d(X, pool_size, mode = 'max'):
    pool_h, pool_w = pool_size
    Y = torch.zeros((X.shape[0] - pool_h + 1, X.shape[1] - pool_w + 1)) # 结果矩阵
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i : i + pool_h, j : j + pool_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i : i + pool_h, j : j + pool_w].mean()

    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2), mode='avg'))
print(pool2d(X, (2, 2), mode='max'))


# 直接调用
X = torch.arange(16, dtype = torch.float32).reshape((1, 1, 4, 4))
print(X)

pool2d_package = nn.MaxPool2d(3)
print(pool2d_package(X)) # tensor([[[[10.]]]])

# 手动调节padding和strid
pool2d_2 = nn.MaxPool2d(3, padding = 1, stride = 2)
print(pool2d_2(X))


# 设定一个任意大小的矩形池化窗口， 并分别设定填充和步幅的高度宽度
pool2d = nn.MaxPool2d((2, 3), padding = (1, 1), stride = (2, 3))
pool2d(X)
