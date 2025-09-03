'''
1.什么叫做门控：
    不是每个观察值都是同等重要，句子中也是，关键词比较重要
    想要记住相关的观察，那么需要能关注的机制：（更新门）以及能遗忘的机制（重置门）
    对于普通RNN来说，所有东西都一样重要，但是GRU门控循环网络，就可以关注到哪个更重要，哪个更不重要

2. 什么叫 门？
    RNN中的输入是现在的输入Xt和上一个时间步的隐藏状态Ht-1
    重置门Rt ：算“新内容”之前，先决定要不要参考过去。
    更新门Zt ：算完“新内容”之后，决定写多少新、留多少旧。

补充：
什么叫候选状态： 含义：根据当前输入Xt和（经重置门rt过滤后的）旧隐状态ht-1生成的新信息提案。
'''


import torch
from torch import nn
from LM.d2l import load_data_time_machine, RNNModelScratch, train_ch8, try_gpu, RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

#初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    # 定义一个函数，用于生成服从正态分布的随机张量，并乘以0.01进行缩放
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    #生成三组权重和偏置张量，用于不同的门控机制
    def three():
        '''返回三个东西，直接初始化z-gate的wxz，whz，bz'''
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 初始化GRU中的权重和偏置
    # 权重和偏置用于控制更新门
    W_xz, W_hz, b_z = three()  # 相比于RNN，GRU多了这两行
    # 权重和偏置用于控制重置门
    W_xr, W_hr, b_r = three()  # GRU多了这两行
    # 权重和偏置用于计算候选隐藏状态
    W_xh, W_hh, b_h = three()
    # 隐藏状态到输出的权重
    W_hq = normal((num_hiddens, num_outputs))
    # 输出的偏置
    b_q = torch.zeros(num_outputs, device=device)
    # 参数列表中各参数顺序
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    # 遍历参数列表中所有参数
    for param in params:
        # 设置参数的`requires_grad`属性为True，以便进行梯度计算
        param.requires_grad_(True)
    # 返回参数列表中所有参数
    return params


# 定义隐藏状态的初始化函数，因为一开始的👕时间步隐藏状态得不到
def init_gru_state(batch_size, num_hiddens, device):
    # 返回隐藏状态初始化为全零的元组
    return (torch.zeros((batch_size, num_hiddens), device=device),)


#定义GRU门控单元
def gru(inputs, state, params):
    #参数params解包为多个变量，分别表示模型中的权重和偏置
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    # 传入的隐藏状态 state 解包为单个变量 H。
    H, = state
    # 创建一个空列表，用于存储每个时间步的输出
    outputs = []
    # 遍历输入序列中的每个时间步
    for X in inputs:
        #更新门控机制 Z:(B, Hdim) 更新门，决定“保留旧记忆 h_{t-1} 的比例”。W_xz:(I,Hdim), W_hz:(Hdim,Hdim), b_z:(Hdim,)
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        #重置门控机制 R：(B, Hdim) 重置门，决定“在算候选态时，历史要不要参与”
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        #计算候选隐藏状态H_tilda（～）
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        #更新隐藏状态 H
        H = Z * H + (1 - Z) * H_tilda
        #计算输出 Y
        Y = H @ W_hq + b_q
        #将输出添加到列表中
        outputs.append(Y)
    # 将所有输出拼接在一起，并返回拼接后的结果和最终的隐藏状态
    return torch.cat(outputs, dim=0), (H,)


#train
if __name__ == '__main__':
    vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()
    num_epochs, lr = 500, 1
    # #创建gru实例
    # model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
    #                            init_gru_state, gru)
    # train_ch8(model, train_iter, vocab, lr, num_epochs, device)



    '''简洁实现'''
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)