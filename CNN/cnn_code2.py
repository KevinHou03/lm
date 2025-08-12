import torch
from torch import nn
from torch.nn import Parameter
from cnn_code1 import corr2d

'''
【二维卷积/交叉相关 输出尺寸公式】

设：
- H_in  = 输入高度（input height）
- W_in  = 输入宽度（input width）
- K_h   = 卷积核高度（kernel height）
- K_w   = 卷积核宽度（kernel width）
- P_h   = 总填充行数（padding height, 上下之和）
- P_w   = 总填充列数（padding width, 左右之和）
- S_h   = 高度方向步幅（stride height）
- S_w   = 宽度方向步幅（stride width）

则输出尺寸：
H_out = floor( (H_in - K_h + P_h) / S_h ) + 1
W_out = floor( (W_in - K_w + P_w) / S_w ) + 1


【特殊情况】
1. 如果 stride = 1：
   H_out = H_in - K_h + P_h + 1
   W_out = W_in - K_w + P_w + 1

2. SAME padding（保持输出尺寸不变）的条件（stride=1时）：
   P_h = K_h - 1
   P_w = K_w - 1

3. 奇偶卷积核的对称填充分配：
   - 若 K_h 为奇数：上下各填 P_h / 2
   - 若 K_h 为偶数：上填 ceil(P_h / 2)，下填 floor(P_h / 2)
   （宽方向同理）

'''

import torch
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) # 把二维张量补充成（N,C,H,W） -> (1, 1, H, W)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size = 3 , padding = 1)
X = torch.rand(size = (8,8))
print(comp_conv2d(conv2d, X).shape)

'''填充不同的高度和宽度'''
conv2d = nn.Conv2d(1, 1, kernel_size = (5, 3) , padding = (2, 1))
print(comp_conv2d(conv2d, X).shape)
