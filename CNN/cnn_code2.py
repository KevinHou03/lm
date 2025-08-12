import torch
from torch import nn
from torch.nn import Parameter
from cnn_code1 import corr2d

'''填充 padding 与步幅 stride'''
'''
1. 填充ph行和pw列， 输出形状为(nh - kh + ph + 1) x (nw - kw + pw + 1)
2. 当kh为odd， 在上下两侧填充ph/2
    当kh为even， 在上填充ph/2.ceil， 下侧填充ph/2.floor
'''

import torch
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) # 把二维张量补充成（N,C,H,W） -> (1, 1, H, W)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size = 3 , padding = 1)
X = torch.rand(size = (8,8))
print(comp_conv2d(conv2d, X).shape)

'''填充不同的高度和宽度'''
conv2d = nn.Conv2d(1, 1, kernel_size = (5, 3) , padding = (2, 1))
print(comp_conv2d(conv2d, X).shape)
