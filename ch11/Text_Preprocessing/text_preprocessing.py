import collections
import re
from LM.d2l import DATA_HUB,DATA_URL,download

# 下载并存储 'time_machine' 数据集的 URL 和哈希值
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                               '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """Load the time machine dataset into a list of text lines. """
    """将时间机器数据集加载为文本行的列表。"""
    # 打开 'time_machine' 数据集文件，并使用文件对象 f 进行操作
    with open(download('time_machine'), 'r') as f:
        # 读取文件的所有行，并将每行存储在列表 lines 中
        lines = f.readlines()
        # 把不是大写字母、小写字母的东西，全部变成空格
        # 去除非字母字符，并转换为小写
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]

# 读取时间机器数据集，并将结果存储在 'lines' 变量中
lines = read_time_machine()
print(type(lines), len(lines))
# 打印数据集的第一行
# print(lines[0])
# 打印数据集的第11行（索引为10）
# print(lines[10])

def tokenize(lines, token='word'):
    """
    将文本行列表进行分词处理。

    Parameters:
        lines (list): 文本行列表。
        token (str): 令牌类型，可选值为 'word'（默认）或 'char'。

    Returns:
        list: 分词后的结果列表。

    Raises:
        None
    """
    # 如果令牌类型为 'word'
    if token == 'word':
        # 以空格为分隔符将每行字符串拆分为单词列表
        return [line.split() for line in lines]
    # 如果令牌类型为 'char'
    elif token == 'char':
        # 将每行字符串拆分为字符列表
        return [list(line) for line in lines]
    else:
        # 若指定的令牌类型无效，则打印错误信息
        print('错位：未知令牌类型：' + token)
# 对 lines 进行分词处理，使用默认的 'word' 令牌类型
tokens = tokenize(lines)
# 打印前11行的分词结果
for i in range(11):
    # 空列表表示空行
    print(type(tokens), tokens[i])


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化词表对象。

        Parameters:
            tokens (list): 标记列表（默认为 None）。
            min_freq (int): 最小频率阈值，低于该频率的标记将被过滤掉（默认为 0）。
            reserved_tokens (list): 保留的特殊标记列表（默认为 None）。

        Returns:
            None

        Raises:
            None
        """
        # 如果输入的 tokens 为 None，则将其设置为空列表
        if tokens is None:
            tokens = []
        # 如果保留的特殊标记列表 reserved_tokens 为 None，则将其设置为空列表
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计 tokens 中标记的频率，并返回一个包含标记频率的 Counter 对象
        counter = count_corpus(tokens)  # 遍历得到每一个独一无二token出现的次数
        # 根据标记的频率进行排序，并将结果存储在 self.token_freqs 中
        # sorted() 函数使用 counter.items() 作为排序对象，使用标记频率 x[1] 作为排序依据，降序排序
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 设置未知标记索引为 0，构建包含未知标记和保留特殊标记的列表 uniq_tokens
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        # 将频率大于等于 min_freq 且不在 uniq_tokens 中的标记添加到 uniq_tokens 列表中
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        # 初始化索引到标记和标记到索引的空列表和字典
        self.idx_to_token, self.token_to_idx = [], dict()
        # 遍历 uniq_tokens 中的每个标记，将其添加到索引到标记的列表中，并将标记和对应索引存储到标记到索引的字典中
        # 索引值从 0 开始递增，对应于标记在列表中的位置
        for token in uniq_tokens:
            # 将当前标记 `token` 添加到索引到标记的列表 `self.idx_to_token` 的末尾
            self.idx_to_token.append(token)
            # 将当前标记 `token` 和其对应的索引值存储到标记到索引的字典 `self.token_to_idx` 中
            # 索引值是 `self.idx_to_token` 列表的长度减去 1，即标记在列表中的位置索引
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """
        获取词表的长度。

        Parameters:
            None

        Returns:
            int: 词表的长度。

        Raises:
            None
        """
        # 获取词表的长度
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        根据标记获取其对应的索引或索引列表。

        Parameters:
            tokens (str or list): 标记字符串或标记列表。

        Returns:
            int or list: 标记的索引或索引列表。

        Raises:
            None
        """
        # 如果 tokens 不是列表或元组，则返回对应的索引或默认的未知标记索引
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        # 对于输入的标记列表 tokens，逐个调用 self.__getitem__() 方法获取每个标记对应的索引值，并返回索引值的列表
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        根据索引获取对应的标记或标记列表。

        Parameters:
            indices (int or list): 索引或索引列表。

        Returns:
            str or list: 索引对应的标记或标记列表。

        Raises:
            None
        """
        # 如果输入的 indices 不是列表或元组类型，则返回对应索引值处的标记
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # 对于输入的索引列表 indices，逐个取出每个索引值 index，并通过 self.idx_to_token[index] 获取对应的标记值，最后返回标记值组成的列表
        return [self.idx_to_token[index] for index in indices]


def count_corpus(tokens):
    """
    统计标记的频率。

    Parameters:
        tokens (list): 标记列表。

    Returns:
        collections.Counter: 包含标记频率的 Counter 对象。

    Raises:
        None
    """
    # 检查 tokens 是否是一个列表的列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 如果 tokens 是一个列表的列表，则将其展平为一维列表
        tokens = [token for line in tokens for token in line]
    # 使用 collections.Counter 统计标记的频率
    return collections.Counter(tokens)

# 创建一个 Vocab 对象，将标记列表 tokens 作为参数传入，用于构建词表
vocab = Vocab(tokens)
# 获取词表中的前 10 个标记及其对应的索引值，并将其转换为列表进行打印输出
print(list(vocab.token_to_idx.items())[:10])


# 遍历索引列表 [0, 10]
for i in [0,10]:
    # 打印当前索引 i 处的标记（单词）
    print('word:', tokens[i])
    # 获取当前标记在词表中的索引值
    # 打印当前标记在词表中的索引值（对应的索引值或未知标记索引）
    print('indices:',vocab[tokens[i]])

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的标记索引列表和词汇表"""
    # 加载时光机器数据集的文本行
    lines = read_time_machine()
    # 将文本行转换为字符标记列表
    tokens = tokenize(lines, 'word')
    # 构建词汇表
    vocab = Vocab(tokens)
    # 将文本转换为标记索引列表
    corpus = [vocab[token] for line in tokens for token in line]
    # 截断文本长度（若有限制）
    if max_tokens > 0:
        # 如果设置了最大标记数 max_tokens，对标记索引列表 corpus 进行截断，只保留前 max_tokens 个标记
        corpus = corpus[:max_tokens]
    # 返回截断后的标记索引列表 corpus 和词汇表 vocab
    return corpus, vocab

# 载入时光机器数据集的标记索引列表和词汇表
corpus, vocab = load_corpus_time_machine()
# 打印时光机器数据集的标记数和词汇表大小
print(len(corpus), len(vocab))
print(corpus)
'''
tokens：把文本切好后的“标记列表”。
vocab：你自定义的 Vocab 类的实例（“词表”).它统计 tokens 的频率，给每个独立 token 分配一个唯一的整数ID
'''