import torch
from torch import nn

from LM.utils import load_data_fashion_mnist, train_ch6

net = torch.nn.Sequential(
    nn.Conv2d(1, 6, kernel_size = 5),
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(6, 16, kernel_size = 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Flatten(),
    nn.Linear(256, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)


if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.005, 1, 3
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.train()

    train_ch6(net, train_iter, test_iter, num_epochs, lr, device, True)

