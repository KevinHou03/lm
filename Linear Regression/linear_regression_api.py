import numpy as np
import torch
from torch.utils import data
from linear_regression_scratch import synthetic_data
from torch import nn


true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b,1000 )

def load_array(data_arrays, batch_size, is_train = True):
    # 构造一个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size = batch_size, shuffle = is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))


net = nn.Sequential(nn.Linear(2, 1)) # in_feature, out_feature
'''分别对weight和bias进行初始化'''
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
print(net, net[0])

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 先把梯度清零
        l.backward()
        trainer.step()

    l = loss(net(features), labels)  # 在每个 epoch 结束后，对整体训练数据进行一次评估，查看当前模型的训练损失情况。
    print(f'epoch {epoch + 1}, loss {l:f}')





