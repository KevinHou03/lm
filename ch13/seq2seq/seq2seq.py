'''
最早用于翻译的encoder—decoder架构，
encoder是一个RNN，读取输入的句子（可以是双向的）decoder使用另外一个rnn来输出
注意：双向不能做语言模型，但是可以做翻译模型，他是一个encoder，不是一个decoder

在这个结构里，encoder就是没有输出的RNN，而encoder最后时间步的隐藏状态将会被用作decoder的初始隐状态

check the statsquest
'''
