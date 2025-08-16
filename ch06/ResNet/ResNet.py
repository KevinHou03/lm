'''
ResNet（Residual Network）= “带捷径的深网”。核心点是残差连接（skip/shortcut）
让每个块学习 (𝑥)后与输入 x 相加，输出 y=F(x)+x。这样梯度可以直接穿过恒等映射回流
'''