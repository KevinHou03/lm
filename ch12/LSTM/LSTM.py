'''
# ！statsquest复习一下LSTM结构细节！
相较于GRU(重置门，更新门）
LSTM有三个门：
    忘记门：将值朝0减少 Ft
    输入门：决定是不是忽略输入值 It
    输出门：决定是不是使用隐状态 Ot
具体公式：
输入门:   i_t = σ(W_i x_t + U_i h_{t-1} + b_i) 写入多少新信息
遗忘门:   f_t = σ(W_f x_t + U_f h_{t-1} + b_f) 保留多少旧记忆
输出门:   o_t = σ(W_o x_t + U_o h_{t-1} + b_o)
候选细胞: c_tilde = tanh(W_c x_t + U_c h_{t-1} + b_c) 候选“新内容
细胞更新: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c_tilde 更新细胞状态 -> 保留多少旧记忆*旧记忆 + 更新多少新信息*新信息
隐状态:   h_t = o_t ⊙ tanh(c_t)
- i_t, f_t, o_t ∈ [0, 1]（门向量）

补充：
什么叫候选状态： 含义：根据当前输入Xt和（经重置门rt过滤后的）旧隐状态ht-1生成的新信息提案。
'''

import torch
from torch import nn
from LM.d2l import RNNModelScratch, train_ch8, try_gpu, load_data_time_machine, RNNModel

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    #同样定义一个正态分布函数规范输入范围
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    #定义一个辅助函数three，用于生成三个参数：输入到隐藏状态的权重矩阵、隐藏状态到隐藏状态的权重矩阵和隐藏状态的偏置项
    def three():
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    #输入到隐藏状态的权重矩阵W_xi、隐藏状态到隐藏状态的权重矩阵W_hi和隐藏状态的偏置项b_i
    W_xi, W_hi, b_i = three()
    #输入到隐藏状态的权重矩阵W_xf、隐藏状态到隐藏状态的权重矩阵W_hf和隐藏状态的偏置项b_f
    W_xf, W_hf, b_f = three()
    #输入到隐藏状态的权重矩阵W_xo、隐藏状态到隐藏状态的权重矩阵W_ho和隐藏状态的偏置项b_o
    W_xo, W_ho, b_o = three()
    #输入到隐藏状态的权重矩阵W_xc、隐藏状态到隐藏状态的权重矩阵W_hc和隐藏状态的偏置项b_c
    W_xc, W_hc, b_c = three()
    #隐藏状态到输出的权重矩阵W_hq
    W_hq = normal((num_hiddens, num_outputs))
    #生成输出的偏置项b_q
    b_q = torch.zeros(num_outputs, device=device)
    #将所有参数组合成列表params
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    return params

#初始化
def init_lstm_state(batch_size, num_hiddens, device):
    # 返回一个元组，包含两个张量：一个全零张量表示初始的隐藏状态，和一个全零张量表示初始的记忆细胞状态。
    return (torch.zeros((batch_size, num_hiddens), device=device),
           torch.zeros((batch_size, num_hiddens),device=device))

# 实际模型
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    #解包状态元组state，分别赋值给隐藏状态H和记忆细胞状态C
    (H, C) = state
    #空列表用于存储每个时间步的输出
    outputs = []
    for X in inputs:
        #输入门的计算
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        #遗忘门的计算
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        #输出门的计算
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        #新的记忆细胞候选值的计算：使用输入、隐藏状态和偏置项，通过线性变换和tanh函数计算新的记忆细胞候选值
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        #更新记忆细胞状态：将旧的记忆细胞状态与遗忘门和输入门的乘积相加，再与新的记忆细胞候选值的乘积相加，得到新的记忆细胞状态
        C = F * C + I * C_tilda
        #更新隐藏状态：将输出门和经过tanh函数处理的记忆细胞状态的乘积作为新的隐藏状态
        H = O * torch.tanh(C)
        #输出的计算 通过线性变换得到输出
        Y = (H @ W_hq) + b_q
        #将当前时间步的输出添加到列表中
        outputs.append(Y)
    # 将所有时间步的输出在维度0上拼接起来，作为最终的输出结果；
    # 返回最终的输出结果 和 更新后的隐藏状态和记忆细胞状态的元组
    return torch.cat(outputs, dim=0), (H, C)


if __name__ == '__main__':
    #train
    vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()
    num_epochs, lr = 500, 1
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
    # 使用d2l库中的train_ch8函数进行模型的训练，传入模型对象、训练数据迭代器、词汇表、学习率、训练轮数和设备
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)



    # 简洁实现
    num_inputs = vocab_size
    # 使用nn.LSTM创建一个LSTM层，输入特征数量为num_inputs，隐藏单元数量为num_hiddens
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = RNNModel(lstm_layer, len(vocab))
    mode = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)