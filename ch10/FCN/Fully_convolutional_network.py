'''
普通 CNN（如 VGG、ResNet）最后通常有 全连接层（Fully Connected Layer, FC），输出一个固定长度向量，用于分类
把全连接层替换为卷积层（特别是 1×1 卷积），这样网络就能接收 任意大小的输入图像，并输出 对应大小的预测特征图。
预测结果是逐像素的类别分布，因此可以做 语义分割

分为三部分：
1. 下采样（编码器部分）
用标准 CNN（VGG/ResNet backbone），经过卷积 + 池化，不断缩小特征图分辨率，同时提取语义特征。
比如输入 224×224，经过 5 次池化可能变成 7×7

2. 上采样（解码器部分）
用 转置卷积（deconvolution / ConvTranspose） 或 上采样（bilinear interpolation），逐步恢复空间分辨率。输出和输入图像同大小，每个像素都有一个分类概率分布

3. 跳跃连接（Skip Connections）
低层特征保留了更多空间细节，高层特征有更强的语义信息。
FCN 融合它们（比如 FCN-32s, FCN-16s, FCN-8s），提升边缘和细节分割效果

注意：转置卷积/普通卷积与每个像素预测的关系：
普通卷积（Conv2d + Pooling）不断下采样，特征图分辨率越来越小。如果直接在最后预测类别，就只能得到一个全局分类（比如“这是一张猫的图片”），而不能逐像素分类
转置卷积（ConvTranspose2d） 的作用就是 上采样：把低分辨率特征图逐步放大到输入图像大小。最终输出一个和输入同尺寸的特征图，每个位置对应一个像素。
然后在最后一层加一个 1×1 卷积 + softmax，就可以得到 逐像素的类别预测。
'''

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from LM.d2l import torch as d2l

# 使用在ImageNet数据集上预训练的ResNet18模型来提取图像特征
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 使用list函数和children方法列出预训练模型的所有子层（这些子层通常是神经网络的层）
# 然后使用Python的切片语法来取出最后三层
# 这可以帮助我们理解模型的结构，特别是在我们打算对模型进行微调或者使用模型的某些层来提取特征时
print(list(pretrained_net.children())[-3:]) # 查看最后三层长什么样子

# 创建一个全卷积网络实例net
# 使用预训练的ResNet18模型创建一个新的神经网络
# 其中，"*list(pretrained_net.children())[:-2]"这段代码将ResNet18模型的所有子层（除了最后两层）作为新网络的层
# 这样，新网络实际上是ResNet18模型去掉最后两层后的版本
# 这种方法常常用于迁移学习，即利用一个在大型数据集上训练过的模型的特征提取部分，来帮助我们处理新的任务
net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 去掉ResNet18最后两层

# 创建一个形状为(1, 3, 320, 480)的随机张量，这可以看作是一张形状为(320, 480)，有三个颜色通道的图片 这里的3代表图片的颜色通道数量（红、绿、蓝），320和480分别代表图片的高度和宽度
# 随机张量的所有元素都是在[0, 1)之间随机生成的，可以看作是随机图片的像素值
X = torch.rand(size=(1,3,320,480)) # 卷积核与输入大小无关，只有全连接层与输入大小有关
# 将随机生成的图片输入到网络中，通过调用net(X)，进行前向传播
# 打印输出张量的形状，输出的形状通常可以用于检查网络的结构是否正确
# 对于全卷积网络，输出的宽度和高度通常会比输入的小，这是由于卷积和池化操作造成的
# 在这个例子中，输出的宽度和高度应该是输入的1/32，这是由ResNet18的结构决定的
print(net(X).shape)  # 缩小32倍

# 使用1X1卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）
# 将要素地图的高度和宽度增加32倍
# 定义目标数据集中的类别数量，这里的21表示Pascal VOC2012数据集中有21个类别，包括20个物体类别和一个背景类别
num_classes = 21
# 在网络末尾添加一个新的卷积层，这是一个1x1的卷积层，输入通道数为512（这是由前面的ResNet18模型决定的）
# 输出通道数为我们定义的类别数量，即21
# 1x1卷积层常用于改变通道数，即可以将前一层的特征图投影到一个新的空间，这个新的空间的维度即为卷积层的输出通道数
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))
# 图片放大32倍，所以stride为32
# padding根据kernel要保证高宽不变的最小值，16 * 2 = 32，图片左右各padding
# kernel为64，原本取图片32大小的一半，再加上padding的32，就相当于整个图片
# 再添加一个转置卷积层，转置卷积也被称为反卷积，通常用于将小尺寸的特征图放大到原来的大小
# 这里的输入和输出通道数都是num_classes，表示我们希望在放大的过程中保持通道数不变
# kernel_size是64，stride是32，这意味着这一层将特征图的宽度和高度放大了32倍
# padding是16，它用于在特征图的边缘添加额外的区域，使得输出的大小正好是输入的32倍
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))