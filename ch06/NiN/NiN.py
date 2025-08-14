# NiN 的目标是：让“卷积核本身更像一个小网络”，从而提升特征表达力；并用 全局平均池化（GAP） 取代大而易过拟合的全连接层（FC）。

import torch
from torch import nn

from LM.ch06.LeNet.LeNet import evaluate_accracy_gpu
from LM.utils import Accumulator, accuracy, load_data_fashion_mnist


def nin_block(in_channel, out_channel, kernel_size, strides, paddings):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride=strides, padding=paddings),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=1), # 1x1 conv
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=1), # 1x1 conv不会改变输出通道大小
        nn.ReLU(),
    )

net = nn.Sequential(
    nin_block(1, 96, 11, 4, 0),
    nn.MaxPool2d(3, 2),
    nin_block(96, 256, 5, 1, 2),
    nn.MaxPool2d(3, 2),
    nin_block(256, 384, 3, 1, 1),
    nn.MaxPool2d(3, 2),nn.Dropout(0.5),
    nin_block(384, 10, 3, 1, 1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

def train_ch6(net, train_iter, test_iter, num_epoch, lr, device, dry_run = False):
    '''train model with a GPU'''
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training is on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)

    if dry_run:
        for i in range(3):
            net.train()
            X, y = next(iter(train_iter))
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            grad_sum = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
            optimizer.step()
            print(f"[dry_run] loss={l.item():.4f}, grad_sum={grad_sum:.1f}")
        return  # 直接返回，不进入完整训练

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
            metric.add(l.item() * X.size(0), accuracy(y_hat, y), X.size(0))
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accracy_gpu(net, test_iter, device)
        print(f'for epoch {epoch} the loss is {train_l}, train acc is {train_acc}, test_acc is {test_acc}')

if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.05, 1, 3
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=224)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.train()

    train_ch6(net, train_iter, test_iter, num_epochs, lr, device, True)
