import torch
from IPython import display
from LM.Softmax.Accumulator import Accumulator
from LM.Softmax.MNIST_DS import load_data_fashion_mnist

# from MNIST_DS import *


# 加载数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# 返回训练和测试集的迭代器
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.ones(num_outputs, requires_grad=True) # 偏移是加在每个输出上的，所以只要outputs个就习惯了

# 实现softmax： 将输出值转化为概率分布
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

# 测试softmax
'''
X = torch.normal(0, 1, (2, 5))
print(X)
X_prob = softmax(X)
print(X_prob)
X_prob = X_prob.sum(1)
print(X_prob) # 1, 1
'''

# 一个针对softmax的简单网络
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

'''
# 根据每个样本的真实类别标签，从预测概率中选出对应类别的概率值
y = torch.tensor([0, 2]) # 第一个样本的真实标签是类别0, 第二个是2
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 这里表示第一个和第二个样本对0/1/2的预测概率分别是多少
print(y_hat[[0, 1], y]) # tensor([0.1000, 0.5000])
'''


# 交叉墒cross entropy： 计算模型预测的概率和真实标签之间的差距
def cross_entropy(y_hat, y):
    '''
    :param y_hat: 形状为 [batch_size, num_classes] 的张量，表示每个样本对每个类别的预测概率（已经 softmax 过了）
    :param y: 形状为 [batch_size] 的整数张量，表示每个样本的真实类别标签
    :return:
    '''
    return -torch.log(y_hat[range(len(y_hat)), y])

# print(cross_entropy(y_hat, y))

'''
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.3, 0.2, 0.5]])  # 2个样本，3个类别

y = torch.tensor([2, 0])  # 样本0的真实类别是2，样本1的真实类别是0

y_hat[range(2), y] → y_hat[0][2], y_hat[0][0] -> tensor([0.6, 0.3]) -> -log([0.6, 0.3]) # 分别取出每个样本预测的正确类的概率
'''

# 接下来实现accuracy计算: 计算一个 batch 中预测的准确值的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y # cmp: True 表示预测正确，False 表示预测错误。
    # y_hat.type(y.dtype)是把yhat的类型转化为y的数据类型
    return float(cmp.type(y.dtype).sum())

# print(accuracy(y_hat, y) / len(y))

# 计算整个数据集上的准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精确程度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# print(evaluate_accuracy(net, train_iter))


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3) # [total loss, total accurate count, total sample number]
    for X, y in train_iter:
        y_hat = net(X) # output
        l = loss(y_hat, y) # loss
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad() # as usual, clean up all previous value
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
            # float(l) * len(y)是因为l是平均损失， 需要算出总量才行
        else: # 自定义的梯度下降函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l), accuracy(y_hat, y), y.size().numel())

    return metric[0] / metric[2], metric[1] / metric[2] # 分别返回所有的loss / 样本总数， 和所有的分类正确数 / 样本总数


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):

    train_metrics = (0.0, 0.0)
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

    train_loss, train_acc = train_metrics
    print("train loss:", train_loss)
    print("train_acc:", train_acc)
    return train_loss, train_acc

if __name__ == "__main__":
    # 临时测试代码，只在直接运行这个脚本时才触发
    from LM.Softmax.MNIST_DS import load_data_fashion_mnist
    batch_size = 256
    train_iter, _ = load_data_fashion_mnist(batch_size=batch_size)
    from LM.MLP.mlp_scratch import net  # 或你临时定义一个 net
    print(evaluate_accuracy(net, train_iter))
