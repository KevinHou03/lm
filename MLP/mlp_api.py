import torch
from torch import nn

from LM.Softmax.MNIST_DS import load_data_fashion_mnist
from LM.Softmax.softmax_scratch import train_ch3

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(256, 10))

def init_weights(m): # 初始化神经网络的参数
    '''
    :param m:表示网络中的某一层模块（比如 nn.Linear、nn.Conv2d 等）。
    :return:
    '''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights) # 没有显式传参，但是内部会自动把网络的每一层都给他初始化

if __name__ == "__main__":
    batch_size, lr, num_epochs = 1, 0.1, 1
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.Adam(net.parameters(), lr=lr)

    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_loss, train_acc = train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    print(train_loss, train_acc)

'''
# 常见权重初始化方法（适用于 m.weight）：
# 方法             | 函数名                          | 说明
# ----------------|----------------------------------|----------------------------
# 正态分布         | nn.init.normal_                 | mean=0, std=0.01
# 均匀分布         | nn.init.uniform_                | a=-0.1, b=0.1
# Xavier 初始化    | nn.init.xavier_uniform_         | 适用于 tanh 激活
# Kaiming He 初始化| nn.init.kaiming_normal_         | 适用于 ReLU 激活

# 示例用法：
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)

'''