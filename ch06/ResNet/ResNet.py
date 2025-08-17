'''
ResNet（Residual Network）= “带捷径的深网”。核心点是残差连接（skip/shortcut）
让每个块学习 (𝑥)后与输入 x 相加，输出 y=F(x)+x。这样梯度可以直接穿过恒等映射回流
'''

import torch
from torch import nn
from torch.nn import functional as F

from LM.utils import load_data_fashion_mnist, train_ch6

"""
1×1 卷积的核心作用是对“通道维度”做可学习的线性变换：在不改变空间尺寸（stride=1, padding=0）的情况下，
对每个像素位置按通道进行重组。主要用途：
① 通道变换（升/降维），在 ResNet 的投影捷径中用 1×1 将输入的通道数和步幅对齐主分支，保证能做 F(x)+x；
② 瓶颈降算力，先用 1×1 将通道从 C 降到 C′ 再做 3×3 卷积，参数/算力显著降低（如 C=256、C′=64 时约可节省 8× 以上）；
③ 跨通道特征重混合，匹配 BN/激活后提升表达力（Network-in-Network 思想）；
④ 形状对齐/下采样，设置 stride=2 的 1×1 在捷径里同步下采样并匹配通道；
⑤ Inception 中的降维，先 1×1 降维再接 3×3/5×5 分支以实现多尺度且省算力。要点：1×1 不扩展感受野，
但能高效重排通道并配合非线性提升表示能力，是 ResNet/GoogLeNet 等架构中的关键积木。
"""
class Residual(nn.Module):
    def __init__(self, in_channel, num_channel, use_1x1conv=False, strides=1):
        '''
        :param in_channel:输入通道数
        :param num_channel:块内输出通道数
        :param use_1x1conv:是否用 1×1 投影捷径把x的形状变成与残差分支一致
        :param strides:
        '''
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, num_channel,
                               kernel_size=3,
                               stride=strides,
                               padding=1)
        self.conv2 = nn.Conv2d(num_channel, num_channel,
                               kernel_size=3,
                               padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channel, num_channel,
                                   kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.relu = nn.ReLU(inplace=True) #inplace 原地操作，节约内存

    def forward(self, X):
        Y= self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X= self.conv3(X)
        Y += X
        return self.relu(Y)

# 输入输出形状一致
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
# print(Y.shape) # torch.Size([4, 3, 6, 6])


#增加输出通道数的同时，减半输出的H和W
blk = Residual(3, 6, use_1x1conv=True, strides = 2)
# print(blk(X).shape) # torch.Size([4, 6, 3, 3])
'''
为什么要通道数翻倍，HW减半：
更大感受野：下采样（stride=2）让后续卷积看到更大的上下文。
表示能力不掉：虽然 H×W 变小了，但把 通道数翻倍 增强通道维的表达，使总信息量不至于下降。
算力均衡：空间减半（≈1/4 像素数），通道翻倍（×2），总体张量元素数约减半；卷积的 FLOPs 在各 stage 之间更平衡。
'''

# ResNet的第一个stage， 快速下采样（把分辨率直接降到原来的 1/4），把通道提到 64，提取低级纹理。
b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                  nn.BatchNorm2d(64),nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

# class Residual为小block，resnet_block 为大block，为Resnet网络的一个stage
def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block: # 每个 stage 的第一个块（且 first_block=False 时）在减半。
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 因为resnet_block返回的是一个[]所以用*解包-> nn.Sequential(*[m1, m2, m3]) ⟹ nn.Sequential(m1, m2, m3)
b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True)) # 因为b1做了两次宽高减半，nn.Conv2d、nn.MaxPool2d，所以b2中的首次就不减半了
b3 = nn.Sequential(*resnet_block(64,128,2)) # b3、b4、b5的首次卷积层都减半
b4 = nn.Sequential(*resnet_block(128,256,2)) # 看一下这个是什么意思
b5 = nn.Sequential(*resnet_block(256,512,2))


net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))


# 观察一下ReNet中不同模块的输入形状是如何变化的
X = torch.rand(size=(1,1,224,224))
for layer in net:
    X = layer(X)
    # print(layer.__class__.__name__,'output shape:\t',X.shape) # 通道数翻倍、模型减半


if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.0005, 1, 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.train()

    train_ch6(net, train_iter, test_iter, num_epochs, lr, device, True)

