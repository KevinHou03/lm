import torch
from torch import nn
from LM.ch02.Softmax.MNIST_DS import load_data_fashion_mnist
from LM.ch02.Softmax.softmax_scratch import train_ch3

# 设置参数
num_inputs, num_outputs, num_hidden = 784, 10, 256
'''
这里的num-hidden是表示隐藏层中一共有256个神经元，每个神经元会需要一个b-weight
requires_grad 为 True 表示 该张量是可训练的参数，会被 PyTorch_TuDui 的 autograd 自动记录
'''
W1 = nn.Parameter(torch.randn(num_inputs, num_hidden, requires_grad=True)) # 模型的可学习参数
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hidden, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)

params = [W1, b1, W2, b2]

# relu
def relu(X):
    a = torch.zeros_like(X) # zeros_like 用于创建一个 形状和数据类型与指定张量 input 相同，但值全为 0 的新张量
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2

if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size = batch_size)

    loss = nn.CrossEntropyLoss()
    num_epochs, lr = 3, 0.01
    updater = torch.optim.SGD(params, lr = lr)

    train_loss, train_acc = train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    print(train_loss, train_acc)

