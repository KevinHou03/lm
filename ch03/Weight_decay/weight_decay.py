import matplotlib as plt
import numpy as np
import torch
from torch import nn




# 人造数据集： labels=X⋅true_w+true_b+noise
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs)) # 	生成 120 个样本（合并训练+测试），每个是 200 维标准正态分布向量
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype=torch.float) # 添加噪声
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 生成一个随机训练数据集
num_epochs, lr = 100, 0.003
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size = (num_inputs, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad = True) # 线性回归共享一个b
    return [w, b]

# L2正则化
def L2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 定义一个简单的网络和一个loss
def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义一个简单的小批量梯度下降
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size
        param.grad.zero_()

def train_no_api(lambd):
    w, b = init_params() #首先初始化
    net, loss, updater = lambda X: linreg(X, w, b), squared_loss, torch.optim.SGD
    # loss可以不用手写：loss_fn = torch.nn.MSELoss()
    num_epoch, lr = 100, 0.003

    for epoch in range(num_epoch):
        for X, y in train_iter:
            with torch.enable_grad():
                # 这里就是如何实现l2正则化
                l = loss(net(X), y) + lambd * L2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
            # optimizer = torch.optim.SGD([w, b], lr=0.003, Weight_decay=lambd)
    print("w' s L2 is ", torch.norm(w).item())

def train_api(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    num_epoch, lr = 100, 0.003
    for param in net.parameters():
        param.data.normal_() # 将所有参数（权重、偏置）用标准正态分布（均值0，方差1）初始化
    loss = nn.MSELoss()
    #  对权重使用 L2 正则, 对b则不使用
    updater = torch.optim.SGD([{'params': net[0].weight, 'Weight_decay': wd}, {'params': net[0].bias}], lr=lr, momentum=0.9)

    for epoch in range(num_epoch):
        for X, y in train_iter:
            with torch.enable_grad():
                updater.zero_grad()
                l = loss(net(X), y)
            l.backward()
            updater.step()

    w_norm = torch.norm(net[0].weight).item()
    print(f"L2 norm of w: {w_norm:.6f}")



if __name__ == '__main__':
    train_no_api(lambd=3)
    train_api(wd=0)
    train_api(wd=3)
    # 将lambd设置为0的时候l2特别大，w也会特别大，因为模型只追求让训练误差最小， 会让权重 w 变得很大以过度拟合训练集中的每个样本



