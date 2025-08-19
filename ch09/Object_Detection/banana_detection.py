import os
import pandas as pd
from matplotlib import pyplot as plt

from LM.d2l import torch as d2lBoook, show_images, show_bboxes
import torch, torchvision

import os, zipfile, urllib.request

DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/"
DATA_DIR = os.path.expanduser("~/.cache/banana")
os.makedirs(DATA_DIR, exist_ok=True)

def download_extract():
    zip_path = os.path.join(DATA_DIR, "banana-detection.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(DATA_URL + "banana-detection.zip", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    return os.path.join(DATA_DIR, "banana-detection")

root = download_extract()
print("dataset root:", root)

def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签（labels 归一化到[0,1]，形状 (N,1,5)）"""
    data_dir = download_extract()  # 例如 ~/.cache/banana/banana-detection
    sub = 'bananas_train' if is_train else 'bananas_val'
    csv_fname = os.path.join(data_dir, sub, 'label.csv')

    df = pd.read_csv(csv_fname).set_index('img_name')
    images, labels = [], []

    for img_name, row in df.iterrows():
        img_path = os.path.join(data_dir, sub, 'images', img_name)
        img = torchvision.io.read_image(img_path).to(torch.float32) / 255.0  # [C,H,W]
        _, H, W = img.shape

        # row: [label, xmin, ymin, xmax, ymax]（像素）
        cls  = float(row['label'])
        xmin = float(row['xmin']) / W
        ymin = float(row['ymin']) / H
        xmax = float(row['xmax']) / W
        ymax = float(row['ymax']) / H

        images.append(img)
        labels.append([cls, xmin, ymin, xmax, ymax])

    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N,1,5)
    return images, labels

'''
images：形状通常是[batch_size, 3, 256, 256]的张量(B×C×H×W，数值在 0~1）。
labels：形状是 [batch_size, 1, 5]，每张图 1 个框，5 个数是 (class, xmin, ymin, xmax, ymax)（你已做 0~1 归一化）。
'''


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train: bool):
        self.features, self.labels = read_data_bananas(is_train)
        print(f"read {len(self.features)} " + ("training examples" if is_train else "validation examples"))

    def __getitem__(self, idx):
        # features: [C,H,W] float32 in [0,1]; labels: [1,5] (cls,xmin,ymin,xmax,ymax)
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size: int):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=True),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_iter, val_iter

# 取一个小批量，并打印其中的图像和标签的形状
batch_size, edge_size = 32, 256 # 香蕉数据集的图片本身就是 256×256。
train_iter, val_iter = load_data_bananas(batch_size)
batch = next(iter(train_iter)) # 从 DataLoader 里取出第一批数据
print(batch[0].shape, batch[1].shape)

imgs = (batch[0][0:10]).permute(0, 2, 3, 1)
axes = show_images(imgs, 2, 5, scale = 2)
for ax, label in zip(axes, batch[1][0:10]):
    show_bboxes(ax, [label[0][1:5] * edge_size], colors = ['w'])
plt.show()