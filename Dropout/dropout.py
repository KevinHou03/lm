import torch
from torch import nn

def dropout_layer(X, dropout_rate):
    assert 0 <= dropout_rate <= 1 # 丢弃率必须是 0 到 1 之间
    if dropout_rate == 1:
        return torch.zeros_like(X)
    if dropout_rate == 0:
        return X
    # 生成一个与X相同形状的张量，元素是 [0, 1) 之间的随机数；float将bool -> 0/1 -> 大概有 1 - dropout_rate 的位置是 1，其余是 0。
    mask = (torch.randn(X.shape)> dropout_rate).float() # 1. randn是生成随机标准正态分布， uniform_(0, 1)把张量就地改写为01之间的均匀分布
    # 用mask * X将被丢弃的位置置为0
    return mask * X / (1.0 - dropout_rate) # 1 - dropout 保持期望不变

# test
X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 0.0))
print(dropout_layer(X, 1.0))