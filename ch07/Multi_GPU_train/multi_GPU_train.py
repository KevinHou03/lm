import torch
from torch import nn
from torch.nn import functional as F

from LM.ch05.GPU.gpu import try_gpu
from LM.utils import load_data_fashion_mnist

# 简单网络 lenet
scale = 0.01
W1 = torch.randn(size=(20,1,3,3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50,20,5,5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800,128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128,10)) * scale
b4 = torch.zeros(10)
params = [W1,b1,W2,b2,W3,b3,W4,b4]

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0],bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2,2), stride=(2,2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation,kernel_size=(2,2),stride=(2,2))
    h2 = h2.reshape(h2.shape[0],-1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

loss = nn.CrossEntropyLoss(reduction='none')

'''给你参数params，全部移到device：GPU上'''
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()# 把这个张量标记为需要梯度，优化器才有东西可更新
    return new_params

'''
在所有的gpu上有一些data，把他们放到同一个gpu上加起来（此操作只能在同一个gpu上使用），
把一个张量列表 data（可能分布在不同设备上）的值相加到 data[0]，然后把结果拷回每个元素，
实现“所有副本都拿到和”的简易 all-reduce。
'''
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device) # 把结果赋给其他位置

'''把X和y拆分到多个设备上'''
def split_batch(X, y, devices):
    assert X.shape[0] == y.shape[0]
    return nn.parallel.scatter(X, devices), nn.parallel.gather(y, devices)

def build_optimizers(device_params, lr):
    # 每个设备一份参数 -> 每个设备一个优化器
    return [torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
            for params in device_params]

'''在多GPU上训练一个mini batch'''
def train_batch(X, y, device_params, devices, lr):
    optimizers = build_optimizers(device_params, lr)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    X_shards, y_shards = split_batch(X, y, devices)
    # 每卡求“样本和”的 loss，保证梯度按样本数加和
    ls = [loss(lenet(X_s, W_s), y_s).sum()
          for X_s, y_s, W_s in zip(X_shards, y_shards, device_params)]
    for l in ls:
        l.backward()
    # 对每一层做allreduce(SUM)
    with torch.no_grad():
        num_layers = len(device_params[0])
        for i in range(num_layers):
            grads_i = [device_params[c][i].grad for c in range(len(devices))]
            allreduce(grads_i)
        # 把SUM变成“按总batch 的平均”再 step
        total_bs = X.shape[0]
        for params, opt in zip(device_params, optimizers):
            for p in params:
                if p.grad is not None:
                    p.grad.div_(total_bs)
            opt.step()

def train(num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)] # 创建多少个GPU
    device_params = [get_params(params, d) for d in devices] # 初始化的params复制到每个GPU上
    num_epochs = 10
    for epoch in range(num_epochs):
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize() # 等待所有的GPU算完
