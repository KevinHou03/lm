'''
一个变圆框通过四个数字来定义，左上x/y 右下x/y，或者左上x/y和宽/高
关于dataset: 每行表示一个物体，文件名-类别-变圆框

物体检测识别图片里的多个物体的类别和位置，位置通常用变圆框表示
'''


import torch
import matplotlib.pyplot as plt
from PIL import Image


torch.set_printoptions(2) # 将打印的张量的精度设置为2位小数
help(torch.set_printoptions) # 将打印的张量的精度设置为2位小数

def set_figsize(size = (4.5, 3.5)):
    plt.rcParams['figure.figsize'] = size

img_path = "/Users/kevinhou/Documents/cats/BunnyHouProfile/Bunny's Pictures/WechatIMG288.jpg"
# 用PIL读图
img = Image.open(img_path).convert("RGB")
plt.imshow(img)
plt.show()

# 定义两种表示函数
# boxes 是一个形状为 (N, 4) 的 torch.Tensor，每一行是一个目标的左上/右下坐标：[x1, y1, x2, y2]
def box_corner_to_center(boxes):
    '''从左上右下 转换到 （中间，宽，高）'''
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim = -1)
    return boxes

def box_center_to_corner(boxes):
    '''从（中间，宽，高）转换为（左上，右下） '''
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim = -1)
    return boxes

# test
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400, 112, 655, 190]
boxes = torch.tensor((dog_bbox, cat_bbox)).float()
print(box_center_to_corner((box_corner_to_center(boxes))) == boxes) # true

# 给定定点坐标，转成框
def bbox_to_rect(bbox, color):
    return plt.Rectangle(xy = (bbox[0], bbox[1]),
                         width = bbox[2] - bbox[0],
                         height = bbox[3] - bbox[1],
                         fill = False,
                         edgecolor = 'red',
                         linewidth = 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img)
ax.add_patch(bbox_to_rect(dog_bbox, color='blue'))
ax.add_patch(bbox_to_rect(cat_bbox, color='red'))
plt.show()