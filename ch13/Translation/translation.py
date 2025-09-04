import os
import torch
from matplotlib import pyplot as plt

from LM.d2l import torch as d2l, DATA_HUB, DATA_URL, download_extract, set_figsize, Vocab

#download dataset (In French)
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入“英语-法语”数据集"""
    data_dir = download_extract('fra-eng')
    # 读取数据并返回
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])

# preprocess.
def preprocess_nmt(text):
    """preprocess“英语-法语”数据集"""
    # 判断字符是否是特定标点符号并且前一个字符不是空格
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    # 替换特殊字符为空格，转换为小写
    text = text.replace('\u202f', ' ').replace('\xa0',' ').lower()
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char # 在特定标点符号前添加空格
        for i, char in enumerate(text)] # 遍历文本中的每个字符
    return ''.join(out) # 将处理后的字符列表转换为字符串

text = preprocess_nmt(raw_text)
print(text[:75])

#tokenization
def tokenize_nmt(text, num_examples=None):
    """tokenize“英语-法语”数据数据集 """
    # 存储英语和法语的词元序列
    source, target = [], []
    # traverse
    for i, line in enumerate(text.split('\n')):
        # 如果指定了num_examples且超过了指定数量，则结束循环
        if num_examples and i > num_examples:
            break
        # 按制表符分割行，实际上就是按英语-法语分开，因为一行同时有两种语言对应
        parts = line.split('\t')
        # 如果行中包含了两个部分
        if len(parts) == 2:
            # 将英语部分按空格分割为词元，添加到source
            source.append(parts[0].split(' ')) # 英语
            # same for French
            target.append(parts[1].split(' ')) # 法语
    return source, target

# 调用函数词元化文本
source, target = tokenize_nmt(text)
# 打印source和target的前6个词元序列
print(source[:6], target[:6])

# 绘制每个文本根据句子长度做的直方图
# 设置图形大小
set_figsize()
# 绘制每个文本序列所包含的标记数量的直方图
_, _, patches = plt.hist([[len(l)
                              for l in source], [len(l) for l in target]],
                            label = ['source','target']) # 添加标签
# 遍历第二个直方图的每个矩形
for patch in patches[1].patches:
    # 设置矩形的填充样式为斜线
    patch.set_hatch('/')
# 添加图例，位于右上角
plt.legend(loc='upper right')
plt.show()

#构建词汇表
src_vocab = Vocab(source, min_freq=2,
                     reserved_tokens=['<pad>','<bos>','<eos>']) # bos 表示句子开始，eos表示句子结束，min_freq=2表示句子长度小于2个就不要了
# 计算词汇表的大小
print(len(src_vocab))


# 序列样本都有一个固定长度截断或填充文本序列
def truncate_pad(line, num_steps, padding_token):
    # 如果文本序列长度超过了指定的长度
    if len(line) > num_steps:
        # 截断文本序列,只要固定长度
        return line[:num_steps]
    # 如果不够长，填充文本序列，添加padding_token直到长度达到num_steps
    return line + [padding_token] * (num_steps - len(line))
# 对源语言的第一个文本序列进行截断或填充
truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])


# 转换成小批量数据集用于训练
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # 将文本序列中的词元转换为词汇表中的索引
    lines = [vocab[l] for l in lines]
    # 每个句子后面加一个截止符'<eos>'
    lines = [l + [vocab['<eos>']] for l in lines] # 每个句子后面加了一个截止符
    # 构建小批量数据集的张量表示
    array = torch.tensor([ truncate_pad(l, num_steps, vocab['<pad>']) for l in lines ])
    # 计算原始句子的实际长度
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    # 返回小批量数据集的张量表示和实际长度
    return array, valid_len # valid_len 为原始句子的实际长度