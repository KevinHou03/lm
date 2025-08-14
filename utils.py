'''计算批量上的正确率'''
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils import data



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
