import torch
from torch import nn
from torch.nn import Sequential

from LM.Softmax.MNIST_DS import load_data_fashion_mnist

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

net = Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)

net.apply(init_weights)