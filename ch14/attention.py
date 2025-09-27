import torch
from torch import nn
from LM.d2l import *

'''
核回归（Nadaraya–Watson estimator）
query: x 你提出的问题
keys: 训练点 判断哪个信息更与你的问题相关
values: 标签y 从对应的key取出信息value
所以用Query去匹配Keys算出相关度，再用这些相关度对Values做加权平均
'''

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
# 定义函数f，用于生成标签y
def f(x):
    return 2 * torch.sin(x) + x**0.8
# 生成训练集标签y_train，并加上服从正态分布的噪声
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
# 生成测试集特征x_test，范围为[0, 5)，步长为0.1
x_test = torch.arange(0, 5, 0.1)
# 生成测试集的真实标签y_truth
y_truth = f(x_test)
# 计算测试集样本数量
n_test = len(x_test)
print(n_test)

# 绘制核回归结果的图像
def plot_kernel_reg(y_hat):
    plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth','Pred'],
            xlim=[0,5], ylim=[-1,5])
    plt.plot(x_train, y_train, 'o', alpha=0.5)
# 将y_train的均值重复n_test次作为预测标签y_hat
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# 调用plot_kernel_reg函数，绘制核回归结果的图像
plot_kernel_reg(y_hat)
plt.show()

# 非参数注意力pooling
# 将测试集特征x_test重复n_train次并重新reshape为二维矩阵
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# 计算注意力权重，通过对特征差值的平方取负并除以2，再进行softmax归一化
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# 注意力权重与训练集标签y_train进行矩阵乘法得到预测标签y_hat
y_hat = torch.matmul(attention_weights, y_train)
# 调用plot_kernel_reg函数，绘制非参数注意力汇聚的核回归结果图像
plot_kernel_reg(y_hat)