import torch
from torch import nn
from LM.utils import Accumulator
from LM.utils import load_data_fashion_mnist
from LM.utils import accuracy

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
for layer in net.children():
    X = layer(X)
    # print(layer.__class__.__name__, 'output shape:', X.shape)


def evaluate_accracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if device is None:
            device = next(net.parameters()).device

    metric = Accumulator(2)  # 正确数, 总数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, (list, tuple)):     # 只有多输入才按元素搬设备
                X = [x.to(device) for x in X]
                y_hat = net(*X)
            else:                                 # 常见：单输入张量
                X = X.to(device)
                y_hat = net(X)
            y = y.to(device)
            metric.add(accuracy(y_hat, y), y.numel())

    net.train()  # 可选：恢复训练模式
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
        print(f'for epoch {epoch} the loss is {train_l}, train acc is {train_acc}, test_acc is {test_acc}')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epoch = 0.001, 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ch6(net, train_iter, test_iter, num_epoch, lr, device)


