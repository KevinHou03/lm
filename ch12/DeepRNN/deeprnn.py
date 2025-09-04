'''
generally，单层rnn只有一层隐藏状态在滚动
多层rnn，是把单层rnn按层纵向堆叠，而不是横向拼接，横向的话只增加了序列长度，只有纵向才会增加深度
每一层有自己的参数，顶层rnn的隐状态则为输入，上一层的隐藏序列则为下一层的输入
同一时间步内串联多个非线性变换/Cell， 层间加入残差或门控残差，缓解退化
总之就是使用多个隐藏层，可以获得更多的非线性性
'''
import torch
from torch import nn
from LM.d2l import try_gpu, RNNModel, train_ch8, load_data_time_machine

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
#将输入的维度设置为词汇表的大小，用于将词汇表中的词转换为嵌入向量
num_inputs = vocab_size
device = try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)

#train
num_epochs, lr = 500, 2
train_ch8(model, train_iter, vocab, lr, num_epochs, device)