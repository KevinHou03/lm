'''计算批量上的正确率'''
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils import data
from torch import nn



def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y # cmp: True 表示预测正确，False 表示预测错误。
    # y_hat.type(y.dtype)是把yhat的类型转化为y的数据类型
    return float(cmp.type(y.dtype).sum())

# '''test 1'''
# y_hat = torch.tensor([[0.1, 0.6, 0.3],  # 预测类别=1
#                       [0.2, 0.2, 0.6],  # 预测类别=2
#                       [0.8, 0.1, 0.1]]) # 预测类别=0
# y_true = torch.tensor([1, 0, 0])        # 真实标签
# print("Test 1 正确数：", accuracy(y_hat, y_true))


'''计算整个数据集上的正确率'''
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精确程度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

'''定义一个累加器'''
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data_fashion_mnist(batch_size, resize = None):
    '''下载这个数据集，并且把它加载到内存中'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = '/Users/kevinhou/Downloads/pythonProject/LM/datasets', train = True, transform = trans, download=False) # 我将手动提取
    mnist_test = torchvision.datasets.FashionMNIST(root='/Users/kevinhou/Downloads/pythonProject/LM/datasets', train=False,transform=trans, download=False)

    return data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = get_dataloader_worker()), data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = get_dataloader_worker())

def get_dataloader_worker():
    return 4


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


#  a general train method with dry-run test
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
