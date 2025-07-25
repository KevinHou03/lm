import random
import torch
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # 加入随机噪音
    return X, y.reshape((-1, 1)) # 作为一个列向量返回作为label

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


print('features:',features[0], '\nlabels:', labels[0])
'''features[:, 1] 是一个tensor，所以要先用detach从计算图中分离，再转为numpy'''
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1) # 样本中第2个特征features[:, 1] vs labels
plt.show()

'''从全部数据中按batch_size随机采样小批量样本，为mini-batch训练做准备'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # samples are randomly collected without order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)] # 用min函数防止取样时溢出
        batch_indices = torch.tensor(batch_indices)

        '''yield 表示在函数执行中暂停并返回一个值，但保留当前函数的执行状态'''
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# test
w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

def linear_regression(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    '''
    :param params: 所有需要调整的参数
    :param lr: 学习率
    :param batch_size: 当前批次的样本数 -> 用于平均样本数
    在使用小批量梯度下降mini-batch SGD时，我们通常将梯度除以 batch_size 来对样本求平均，
    这是为了使参数更新的幅度与批量大小无关，从而保持学习率的稳定性和数值稳定性。平均梯度还更好地逼近全数据集的真实梯度，
    有助于模型更平稳地收敛。如果不进行平均，batch_size越大，累积梯度越大，模型更新过快，可能导致训练不稳定。因此，平均梯度是训练中常见且必要的做法
    '''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size # 求一个均值
            param.grad.zero_()


lr = 0.02
num_epochs = 4
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss = squared_loss(linear_regression(X, w, b), y)
        loss.sum().backward() # 反向传播只能用于一个scalar value，所以求和
        sgd([w, b], lr, batch_size)

    with torch.no_grad(): # 这个阶段我们只是做模型评估，不需要计算梯度或构建计算图
        train_loss = squared_loss(linear_regression(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean().item())}')
