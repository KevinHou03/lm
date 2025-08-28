import torch
from torch import nn
from torch.nn import functional as F
from LM.d2l import load_data_time_machine, try_gpu, predict_ch8, train_ch8

# 设置批量大小和时间步数
batch_size, num_steps = 32, 35
# 调用 load_data_time_machine 函数加载时间机器数据集，返回训练数据迭代器和词汇表
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# 定义模型
# 设置隐藏单元的数量为 256
num_hiddens = 256
# 使用 nn.RNN 类定义一个循环神经网络层
# 输入大小为词汇表的大小，隐藏单元数量为 num_hiddens
# 将该循环神经网络层赋值给 rnn_layer
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 使用张量来初始化隐藏状态
# 创建一个形状为 (1, batch_size, num_hiddens) 的张量，用于初始化隐藏状态
# 全部元素初始化为 0
# 将该张量赋值给变量 state
state = torch.zeros((1, batch_size, num_hiddens))
# 打印隐藏状态张量的形状
print(state.shape)

# 通过一个隐藏状态和一个输入，我们可以用更新后的隐藏状态计算输出
# 创建一个形状为 (num_steps, batch_size, len(vocab)) 的随机张量 X
# 用于表示输入的序列，每个时间步的输入为一个词汇表大小的独热编码向量
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
# 将输入 X 和初始隐藏状态 state 作为输入传递给循环神经网络层 rnn_layer 进行前向计算
# 返回输出张量 Y 和更新后的隐藏状态 state_new
Y, state_new = rnn_layer(X, state)
# 打印输出张量 Y 和更新后的隐藏状态 state_new 的形状
print(Y.shape, state_new.shape)


# 我们为一个完整的循环神经网络模型定义一个RNNModel类
class RNNModel(nn.Module):
    """循环神经网络模型"""

    # 初始化函数
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        # 调用父类的构造函数，初始化继承的属性
        super(RNNModel, self).__init__(**kwargs)
        # 循环神经网络层
        self.rnn = rnn_layer
        # 词汇表大小
        self.vocab_size = vocab_size
        # 隐藏状态的大小
        self.num_hiddens = self.rnn.hidden_size
        # 如果循环神经网络不是双向的
        if not self.rnn.bidirectional:
            # 方向数量为1
            self.num_directions = 1
            # 线性层的输入大小为隐藏状态大小，输出大小为词汇表大小
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        # 如果循环神经网络是双向的
        else:
            # 方向数量为2
            self.num_directions = 2
            # 线性层的输入大小为隐藏状态大小的两倍，输出大小为词汇表大小
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

            # 前项传播函数

    def forward(self, inputs, state):
        # 将输入的索引序列转换为独热编码张量 X
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        # 将 X 转换为 float32 类型
        X = X.to(torch.float32)
        # 使用循环神经网络层 rnn 进行前向计算，返回输出张量 Y 和更新后的隐藏状态 state
        Y, state = self.rnn(X, state)
        # 将输出张量 Y 展平并通过线性层 linear 进行变换得到最终的输出
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        # 返回最终输出和更新后的隐藏状态
        return output, state

    # 创建循环神经网络的初始隐藏状态
    def begin_state(self, device, batch_size=1):
        # 如果循环神经网络不是LSTM类型
        if not isinstance(self.rnn, nn.LSTM):
            # 创建全零的隐藏状态张量
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        # 如果循环神经网络是LSTM类型
        else:
            # 创建全零的隐藏状态张量和记忆单元张量
            # 第一个张量是全零的隐藏状态张量，第二个张量是全零的记忆单元张量
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))


# 用一个具有随即权重的模型进行预测
# 尝试使用GPU设备，如果不可用则使用CPU
device = try_gpu()
# 创建RNN模型实例
net = RNNModel(rnn_layer, vocab_size=len(vocab))
# 将模型移动到指定设备上
net = net.to(device)
# 对模型进行预测，生成文本
predict_ch8('time traveller', 20, net, vocab, device)

# 使用高级API训练模型
# 设置训练的迭代周期数和学习率
num_epochs, lr = 500, 1
# 使用高级API训练模型，传入模型、训练数据迭代器、词汇表、学习率和迭代周期数进行训练
train_ch8(net, train_iter, vocab, lr, num_epochs, device)