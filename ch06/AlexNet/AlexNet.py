import torch
from torch import nn
from LM.utils import load_data_fashion_mnist
from LM.utils import Accumulator, accuracy

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.LazyLinear(4096),  # 自动推断 in_features
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 10),
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    # print(layer.__class__.__name__, X.shape)
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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True)
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
            metric.add(l.item() * X.size(0), accuracy(y_hat, y), X.size(0))
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accracy_gpu(net, test_iter, device)
        print(f'for epoch {epoch} the loss is {train_l}, train acc is {train_acc}, test_acc is {test_acc}')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    batch_size = 12
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize = 224)

    lr, num_epoch = 0.001, 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ch6(net, train_iter, test_iter, num_epoch, lr, device)