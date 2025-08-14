import torch
from torch import nn
from torch.nn import Parameter


def corr2d(X, K):
    '''计算二维相关'''
    '''
    :param X: 输入矩阵
    :K: kernel
    '''
    h, w = K.shape # kernel size
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i : i + h, j : j + w] * K).sum() # 取出输入中与核同大小的窗口: X[i : i + h, j : j + w]
    return Y

# test
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


'''实现卷积层'''

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) * self.bias


# test, 检测图像中不同颜色的边缘
X = torch.ones((6, 8))
X[: , 2:6] = 0
print(X)
'''
[1., 1., 0., 0., 0., 0., 1., 1.],
[1., 1., 0., 0., 0., 0., 1., 1.],
[1., 1., 0., 0., 0., 0., 1., 1.],
[1., 1., 0., 0., 0., 0., 1., 1.],
[1., 1., 0., 0., 0., 0., 1., 1.],
[1., 1., 0., 0., 0., 0., 1., 1.]])
'''
# set kernel
K = torch.tensor([[1.0, -1.0]]) # 两边元素没有变化的时候输出为0， 有变化就为1/-1

# result
Y = corr2d(X, K)
print(Y)
'''
[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
[ 0.,  1.,  0.,  0.,  0., -1.,  0.]
'''

'''学习由X生成Y的卷积核'''
conv2d = nn.Conv2d(1, 1, kernel_size = (1,2), bias = False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
optimizer = torch.optim.SGD(conv2d.parameters(), lr=0.03)

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad # 手动更新
    # optimizer.step() # 自动更新

    if(i + 1) % 2 == 0:
        print(f'batch{i + 1}, loss {l.sum():.3f}')


'''所学到的卷积核的权重tensor'''
print(conv2d.weight.data.reshape((1, 2))) # tensor([[ 0.9617, -1.0171]])