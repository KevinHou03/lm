'''
最早用于翻译的encoder—decoder架构，
encoder是一个RNN，读取输入的句子（可以是双向的）decoder使用另外一个rnn来输出
注意：双向不能做语言模型，但是可以做翻译模型，他是一个encoder，不是一个decoder
'''