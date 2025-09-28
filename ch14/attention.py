'''
注意力分数是query和key的相似度，注意力权重是分数的softmax结果
两种常见的分数计算为：
1. 将query和key合并起来进入一个单输出隐藏层的mlp
2. 直接将query和key做内积

=======================
注意力机制：分数、权重、池化 (Q/K/V)
=======================

符号说明:
  Q: 查询 (Queries)，形状 [B, Lq, d_k]
  K: 键   (Keys)，   形状 [B, Lk, d_k]
  V: 值   (Values)， 形状 [B, Lk, d_v]
  B: 批大小，Lq: 查询数量，Lk: 键数量
  d_k: 键/查询向量维度，d_v: 值的维度

目标: 对于每个查询位置 i，计算它与所有键 j 的相关度分布，
      然后利用这个分布对对应的值进行加权汇聚，得到输出表示。

1) 注意力分数 (Attention Scores)
---------------------------------------------------------------
衡量每个查询 q_i 与每个键 k_j 的相关性。

标准缩放点积注意力 (Transformers 中最常用):
  score(q_i, k_j) = (q_i · k_j) / sqrt(d_k)

其他打分函数:
  加性注意力 (Additive/Bahdanau):
    score(q_i, k_j) = v^T tanh(W_q q_i + W_k k_j + b)
  高斯核 (RBF) 形式:
    score(q_i, k_j) = - ||q_i - k_j||^2 / (2 * h^2)   # h = 带宽

数值稳定技巧 (softmax 之前):
  S_ij = score(q_i, k_j)
  S_ij <- S_ij - max_j(S_ij)     # 每行减去最大值避免溢出

2) 注意力权重 (Attention Weights)
-----------------------------------------------------
通过 softmax 将分数转化为概率分布 (按行归一化):
  α_ij = exp(S_ij) / Σ_j exp(S_ij)

性质:
  α_ij >= 0，且对每个查询 i，Σ_j α_ij = 1

掩码 (mask，可选，在 softmax 前应用):
  - Padding mask: 把无效位置设为 -inf，使得 α_ij = 0
  - Causal mask: 约束只能看到当前位置之前的 tokens

3) 注意力池化 (Attention Pooling)
-----------------------------------------------------
利用注意力权重对值 V 做加权平均：
  out_i = Σ_j α_ij * v_j

矩阵形式 (单批次):
  A = softmax(S)          # A 形状: [Lq, Lk]
  Output = A @ V          # 形状: [Lq, d_v]

完整批次形式:
  Scores:  S = (Q @ K^T) / sqrt(d_k)   # [B, Lq, Lk]
  Weights: A = softmax(S, dim=-1)      # [B, Lq, Lk]
  Output:  O = A @ V                   # [B, Lq, d_v]

4) 多头注意力 (Multi-Head Attention, MHA)
-----------------------------------
把 d_model 拆成 h 个头，每个头的维度 d_k = d_model / h:
  对每个头 r:
    Q_r = Q W_q^r,  K_r = K W_k^r,  V_r = V W_v^r
    O_r = softmax((Q_r K_r^T)/sqrt(d_k)) V_r
  拼接所有头并投影:
    O = Concat(O_1, ..., O_h) W_o

优势: 不同的头可以学习到不同的关系模式或尺度。

5) 与核回归 (Nadaraya–Watson) 的关系
------------------------------------------------------
如果选用高斯核分数并 softmax:
  S_ij = - ||q_i - k_j||^2 / (2 h^2)
  α_ij = softmax_j(S_ij)
  out_i = Σ_j α_ij * v_j
这就是一种“核平滑器”，查询会选择周围的键/值。
你的 1D 示例: q = x_test, k = x_train, v = y_train。

6) 伪代码 (概念实现)
------------------------------------------
给定 Q, K, V:
  S = (Q @ K.transpose(-2, -1)) / sqrt(d_k)   # 计算分数
  # 可选: S += 偏置 或 应用 mask (把无效位置设为 -inf)
  A = softmax(S, dim=-1)                      # 得到权重
  O = A @ V                                   # 注意力池化结果
返回 O (以及 A，可用于可视化)

7) 常见陷阱
------------------
- 维度不匹配: 确保 Q,K 的最后一维一致，V 的 Lk 维度对齐。
- 忘记 mask: 在 NLP 里必须对 pad 或未来 token 做屏蔽。
- 数值不稳: softmax 前要减去每行最大值。
- 缩放缺失: 点积注意力必须除以 sqrt(d_k)。

=======================

'''


# 注意力打分函数
import math
import torch
from torch import nn
from LM.d2l import *

# 遮蔽softmax操作，原理就是一行输入有填充值的时候，把他们设置为极小值，这样softmax出来的权重就也是极小值，
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行softmax操作"""
    if valid_lens is None:
        # 如果valid_lens为空，则对X执行softmax操作
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 将valid_lens扩展为与X的最后一个维度相同的形状
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 将valid_lens重塑为一维向量
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # 把每一行在 valid_len 之后的位置用极小值替掉，为后续的 softmax 做遮蔽
        # 对遮蔽后的X执行softmax操作，并将形状还原为原始形状
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# test
masked_softmax(torch.rand(2,2,4), torch.tensor([2,3])) # 1D:同一batch每一个query都用一个有效长度
masked_softmax(torch.rand(2,2,4), torch.tensor([[1,3],[2,4]])) # 2D 每一个(batch,query）对都有一个自己的有效长度


# 加性注意力， 输入→打分→遮蔽→归一化→池化
'''计算注意力分数可以用dot product也可以用addition，这里用的是第二种
将queries和和keys先做各自的线性变换，到相同的num hiddens然后用tanh * 他们的和'''
class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 用于转换键的线性变换
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # 用于转换查询的线性变换
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # 用于生成注意力分数的线性变换
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        # Dropout层，用于随机丢弃一部分注意力权重
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 将查询和键进行线性变换
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 执行加性操作，将查询和键相加
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # 使用双曲正切函数激活加性操作的结果
        features = torch.tanh(features)
        # 使用线性变换生成注意力分数，并将最后一维的维度压缩掉
        scores = self.w_v(features).squeeze(-1)
        # 使用遮蔽softmax计算注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 根据注意力权重对values进行加权求和
        return torch.bmm(self.dropout(self.attention_weights), values)



#showcase
queries, keys = torch.normal(0, 1, (2,1,20)), torch.ones((2,10,2))
values = torch.arange(40, dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)
valid_lens = torch.tensor([2,6])
# 创建加性注意力对象
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
# not using dropout
attention.eval()
# 调用加性注意力对象的forward方法
print(attention(queries, keys, values, valid_lens))


# 缩放点积注意力,用dot product 而不是addition
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        # Dropout层，用于随机丢弃一部分注意力权重
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # 获取查询向量的维度d
        d = queries.shape[-1]
        # 计算点积注意力得分，并进行缩放
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 使用遮蔽softmax计算注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 根据注意力权重对values进行加权求和，这一步叫注意力池化，就是权重和value相乘
        return torch.bmm(self.dropout(self.attention_weights), values)

#showcase
queries = torch.normal(0,1,(2,1,2))
# 创建缩放点积注意力对象
attention = DotProductAttention(dropout=0.5)
attention.eval()
# 调用缩放点积注意力对象的forward方法
attention(queries, keys, values, valid_lens)