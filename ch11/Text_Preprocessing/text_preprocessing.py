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
# 打印数据集的第一行
print(lines[0])
# 打印数据集的第11行（索引为10）
print(lines[10])

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
    print(tokens[i])