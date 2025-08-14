import torch
from torch import nn
from LM.Softmax.Accumulator import Accumulator
from LM.Softmax.MNIST_DS import load_data_fashion_mnist
from LM.Softmax.softmax_api import test_iter
from LM.Softmax.softmax_scratch import accuracy, train_iter

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28) # view是用来改变张量形状的, 把输入的张量 reshape 成 (batch_size, 1, 28, 28)，即单通道的 28×28 图像（MNIST 格式）
    '''
    view(): 速度快、零拷贝（数据连续时）常用在卷积层 → 全连接层之间的 flatten 操作
    reshape(): 更通用，能处理非连续张量（比如经过转置、切片等操作的结果）稍慢一点，但用起来不容易报错
    '''

net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size = 5, padding = 2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(6, 16, kernel_size = 5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size = 2, stride = 2),nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
'''
Flatten扁平化把多维的张量打平成一维向量
因为卷积和池化层输出的是三维结构(num_channel, H, W)(不算batch_size)但是全连阶层要求输入的是二维矩阵(batch_size, features)
所以要用flatten（）拉直。
比如卷积输出的是(batch_size = 2, c = 3, H = 4, W = 4)，flatten之后变成(2, 3*4*4)
'''

X = torch.rand(size = (1, 1, 28, 28), dtype = torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:', X.shape)


def evaluate_accracy_gpu(net, data_iter, device = None):
    '''使用gpu计算模型在数据集上的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(net.parameters()).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epoch, lr, device):
    '''train model with a GPU'''
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training is on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)

    for epoch in range(num_epoch):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_accracy_gpu(net, test_iter, device)

    print(f'loss{train_l}, train acc{train_acc}, test_acc{test_acc}')

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size = batch_size)
lr, num_epoch = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epoch, lr,None)



