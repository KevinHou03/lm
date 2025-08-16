'''
1. batch norm在每一层的一个batch输入上做一个标准化，让数据分布更稳定
2. 把输入变成均值0 方差1
3. 目的：让输入分布更稳定，允许大的学习率， 因为不容易梯度爆炸/消失，减轻过拟合，减少对初始化参数的依赖
4. 使用顺序： conv/mlp -> Batch_Norm -> ReLU(activation) -> pooling/etc.
5. Math: 首先计算一个batch里面的均值：mean = 1/B * (Sum(X_i)) 和方差 variance = 1/B * sum(((X_i - mean)^2))
    归一化：x_norm = (X_i - mean) / sqrt(var + eps)

6. 以上称为标准化，在这个基础上，还要引入可学习的Y和b两个learnable参数： y_i = （Y*x_norm） + b，这样保证batch norm能稳定分布，也不丢失表达能力
'''

# BatchNorm from scratch
import torch
from torch import nn
from LM.utils import load_data_fashion_mnist, train_ch6


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    '''
    :param X:输入张量，可以是 2D (全连接层输出) 或 4D (卷积层输出)。
    :param gamma:BN 的两个 可学习参数，用于线性变换恢复模型的表达能力
    :param beta:BN 的两个 可学习参数，用于线性变换恢复模型的表达能力
    :param moving_mean:在训练中维护的 滑动平均均值与方差（用于推理/inference）
    :param moving_var:在训练中维护的 滑动平均均值与方差（用于推理/inference）
    :param eps:避免除零的数值稳定项。
    :param momentum:滑动平均的更新速率。
    :return:
    '''
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 输入为2为mlp， 输入为4为卷积层，计算维度要调整
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim = 0)
            var = ((X - mean) ** 2).mean(dim = 0)
        else:
            # 所有通道都有一个mean和var，但是在所有batch上共享 -> 按不同通道归一化
            mean = X.mean(dim = (0, 2, 3), keepdim = True)
            var = ((X - mean) ** 2).mean(dim = (0, 2, 3), keepdim = True)
        X_hat = (X - mean) / torch.sqrt(var + eps) # 标准化

        # 更新滑动平均： 更新滑动平均就是把每个 mini-batch 的统计量逐渐融合进一个长期的「全局均值/方差」，训练时更新，推理时使用，保证模型稳定
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    # 线性变换 (恢复模型表达能力)
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

# 创建一个batchnorm图层
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2: # mlp
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 使用nn.Parameter就表明他会记录到model里面参与更新/反向传播等等成为一个可学习的参数，不用的话就不会
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        # # gamma, beta 需要在反向传播时更新，所以放在nn.Parameter里面，moving_mean、moving_var不需要迭代，所以不放在里面
    def forward(self, X):
        # 训练时输入 X 可能在 GPU 上，而 moving_mean 和 moving_var 一开始建在 CPU。
        # 这一步保证它们和 X 在同一个 device 上，否则计算会报错
        if self.moving_mean.device != X.device:
            self.moving_mean = (self.moving_mean.to(X.device))
            self.moving_var = (self.moving_var.to(X.device))
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)  # 每个框架不同eps=1e-5,momentum=0.9 90% 用历史，10% 用当前
        return Y

#将BatchNorm应用于LeNet模型
net = torch.nn.Sequential(
    nn.Conv2d(1, 6, kernel_size = 5),
    BatchNorm(6, 4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(6, 16, kernel_size = 5),
    BatchNorm(16, 4),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(120, 2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, 2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.05, 1, 3
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.train()

    train_ch6(net, train_iter, test_iter, num_epochs, lr, device, True)

    # 查看学习成果
    print(net[1].gamma.reshape((-1, )))
    print(net[1].beta.reshape((-1, )))
