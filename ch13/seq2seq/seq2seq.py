'''
最早用于翻译的encoder—decoder架构，从一个句子生成另一个句子
encoder是一个RNN，读取输入的句子（可以是双向的）decoder使用另外一个rnn来输出，所以两者都是rnn
注意：双向不能做语言模型，但是可以做翻译模型，他是一个encoder，不是一个decoder

在seq2seq里，encoder就是没有输出的RNN，而encoder最后时间步的隐藏状态将会被用作decoder的初始隐状态
seq2seq可以做翻译工作，因为在训练时我们喂给它了“英文句子 → 中文句子”的真实配对。
check the statsquest

1. 训练过程：
假设我们有训练数据：
"I love NLP"   →   "我 爱 NLP"
"Hello world"  →   "你好 世界"
Encoder:
    读入英文句子 "I love NLP"，输出一个语义表示。
Decoder:
    在训练时我们告诉它目标答案 "我 爱 NLP"，然后让它逐词预测：
        - 输入 <BOS>，目标是 "我"
        - 输入 "我"，目标是 "爱"
        - 输入 "爱"，目标是 "NLP"
损失函数:
    对比模型预测和真实目标 (ground truth)，不断更新参数。
这种训练方式叫 teacher forcing。

BLEU定义：机器翻译生成的译文，与一个或多个人工参考译文的 n-gram 重叠程度。越大越好，参考其公示
越长的句子权重越大，
'''


