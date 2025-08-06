import os
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

print('1.访问第二个全连接层的参数')
print(net[2].state_dict())  # 当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层
'''state_dict() 是 PyTorch 中 获取模型所有可学习参数（权重、偏置等）和持久化缓冲区（如 BN 的 running_mean、running_var） 的方法'''
print(net[2].bias)  # 第二个神经网络层提取偏置
print(net[2].bias.data)  # 第二个神经网络层提取偏置的实际值
print(net[2].weight.grad is None)  # 由于我们还没有调用这个网络的反向传播，所以参数的梯度处于初始状态。

print('2.一次性访问所有参数')
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 输入层的参数
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 3.嵌套块的参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


print('3.嵌套块的参数')
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)
print(rgnet(X))
print(rgnet[0][1][0].bias.data)  # 访问第一个主要的块，其中第二个子块的第一层的偏置项

