'''
1. 彩色图片一般有RGB三个通道，黑白图片只有一个通道：灰
2. 每个通道都会有一个卷积核 结果是所有通道卷积结果的和， 所以结果的大小和（卷积核 * 通道）的大小是一样的
3. 无论有多少输入通道，到目前为止我们只用到单通道输出，我们可以有多个三维卷积核，每个核生成一个输出通道。
4. 每个输出通道可以识别一个特定模式，可以理解为有几个输出通道就有几个模式在被识别
5. 输出通道作为下一层的输入，相叠加成一个组合的模式识别。
6. 特殊情况：1 x 1卷积核， 不识别任何空间信息， 但是可以融合不同通道的信息，
'''

import torch

# 处理单通道
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

# 处理多输入多通道二维相关
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))
'''
tensor([[ 56.,  72.],
        [104., 120.]])
'''

# 如果我们想要卷积层一次性产生多个输出通道，就需要为每个输出通道分配一组卷积核
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K],  0)

K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape) # torch.Size([3, 2, 2, 2])

print(corr2d_multi_in_out(X, K))


# 1 x 1 卷积
def corr2d_multi_in_out_1x1(X, K):
