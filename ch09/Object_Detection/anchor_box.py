'''
提出多个被称为anchor box的区域（边缘框）
预测每个box里是否含有关注的物体
如果含有，预测从这个box到真实边缘框的偏移

1. IoU标准（Intersection of Unit）：
用来计算两个框之间的相似度，0表示无重合，1表示重合
计算公式： intersection / union

每个box是一个训练样本，将每个box要么标注成背景，要么关联上一个真实边缘框
生成大量的锚框->生成大量的负类样本

2. NMS(Non-Maximum Suppression，非极大值抑制)：在目标检测里，用来去掉一堆高度重叠的候选框，只保留少数代表性的高分框
先根据分数（置信度）给所有框排序；(去掉冗余）
    1. 取分数最高的框 A；
    2. 计算 A 与其余框的 IoU（交并比：IoU=∣B∪A∣/∣B∩A∣
    3. 把与 A 的 IoU 大于阈值（如 0.5）的框删除；
    4.回到步骤 2，直到没有框为止
'''

import torch
# 从d2l import单个function
from LM.d2l import torch as d2l

