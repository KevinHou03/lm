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

torch.set_printoptions(2)



def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同高宽度的锚框"""
    # data.shape的最后两个元素为宽和高，第一个元素为通道数
    in_height, in_width = data.shape[-2:]
    # 数据对应的设备、锚框占比个数、锚框高宽比个数
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 计算每个像素点对应的锚框数量
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    # 将锚框占比列表转为张量并将其移动到指定设备
    size_tensor = torch.tensor(sizes, device=device)
    # 将宽高比列表转为张量并将其移动到指定设备
    ratio_tensor = torch.tensor(ratios, device=device)

    # 定义锚框中心偏移量
    offset_h, offset_w = 0.5, 0.5
    # 计算高度方向上的步长
    steps_h = 1.0 / in_height
    # 计算宽度方向上的步长
    steps_w = 1.0 / in_width

    # torch.arange(in_height, device=device)获得每一行像素
    # (torch.arange(in_height, device=device) + offset_h) 获得每一行像素的中心
    # (torch.arange(in_height, device=device) + offset_h) * steps_h 对每一行像素的中心坐标作归一化处理

    # 生成归一化的高度和宽度方向上的像素点中心坐标
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 生成坐标网格
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    # 将坐标网格平铺为一维
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 计算每个锚框的宽度和高度
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 计算锚框的左上角和右下角坐标（相对于锚框中心的偏移量）
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 计算所有锚框的中心坐标，每个像素对应boxes_per_pixel个锚框
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    # 通过中心坐标和偏移量计算所有锚框的左上角和右下角坐标
    output = out_grid + anchor_manipulations

    # 增加一个维度并返回结果
    return output.unsqueeze(0)

