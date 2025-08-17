import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from LM.ch05.GPU.gpu import try_all_gpus
from LM.utils import accuracy, Accumulator, evaluate_accracy_gpu

plt.rcParams['figure.figsize'] = (3.5, 2.5)
img = Image.open('/Users/kevinhou/Desktop/截屏2025-08-14 下午10.25.27.png')
plt.imshow(img)
plt.axis('off')
plt.show()

def _to_numpy(img):
    """支持 PIL.Image 或 torch.Tensor -> numpy(H, W, C)。"""
    if isinstance(img, Image.Image):
        arr = np.array(img)  # HWC, uint8
        return arr
    if isinstance(img, torch.Tensor):
        # 形状 [C,H,W] 或 [H,W]; 放到 CPU，转 numpy
        x = img.detach().cpu()
        if x.ndim == 2:  # [H,W] 灰度
            return x.numpy()
        if x.ndim == 3:  # [C,H,W]
            x = x.permute(1, 2, 0)
        if x.dtype.is_floating_point:
            x = x.clamp(0, 1)
        return x.numpy()
    raise TypeError(f"Unsupported image type: {type(img)}")

def show_images(imgs, num_rows, num_cols, scale=1.5):
    """用matplotlib以网格形式显示图片列表imgs。"""
    figsize = (scale * num_cols, scale * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = np.array(axes).reshape(num_rows, num_cols)
    for ax, im in zip(axes.flat, imgs):
        arr = _to_numpy(im)
        # 灰度图用cmap='gray'
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            ax.imshow(arr.squeeze(), cmap='gray')
        else:
            ax.imshow(arr)
        ax.axis('off')
    # 如果张数不足，隐藏多出来的子图
    for ax in axes.flat[len(imgs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 把同一张图片反复做数据增广，然后按网格把结果拼在一张图上展示。
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):  # 传入aug图片增广方法
    Y = [aug(img) for _ in range(num_rows * num_cols)]  # 用aug方法对图片作用八次
    show_images(Y, num_rows, num_cols, scale=scale)

# 水平随机翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
# 上下随机翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
# 随机剪裁，剪裁后的大小为(200,200)
# (0.1,1)使得随即剪裁原始图片的10%到100%区域里的大小，ratio=(0.5,2)使得高宽比为2:1，下面是显示时显示的1:1
apply(img,torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2)))
# 随机更改图像的亮度
apply(img,torchvision.transforms.ColorJitter(brightness=0.8,contrast=0.8,saturation=0,hue=0))
# 随即改变色调
apply(img,torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
# 随机更改图像的亮度（brightness）、对比度（constrast）、饱和度（saturation）和色调（hue）
color_aug = torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)
apply(img,color_aug)
# 结合多种图像增广方法
# 先随即水平翻转，再做颜色增广，再做形状增广
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),color_aug,torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))  ])
apply(img,augs)
'''
# 下载图片，并显示部分图片
all_images = torchvision.datasets.CIFAR10(train=True, root='01_Data/03_CIFAR10', download=True)    
show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
'''

# 只使用最简单的随机左右翻转
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

# 定义一个辅助函数，以便于读取图像和应用图像增广
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='01_Data/03_CIFAR10',train=is_train,
                                         transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train,
                                            num_workers = 0)
    return dataloader

# 定义一个函数，使用多GPU模式进行训练和评估
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X] # 如果X是一个list，则把数据一个接一个都挪到devices[0]上
    else:
        X = X.to(devices[0]) # 如果X不是一个list，则把X挪到devices[0]上
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus()):
    num_batches = len(train_iter)
    # nn.DataParallel使用多GPU
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch_ch13(net,features,labels,loss,trainer,devices)
            metric.add(l,acc,labels.shape[0],labels.numel())
        test_acc = evaluate_accracy_gpu(net,test_iter)
    print(f'loss {metric[0] / metric[2]:.3f}, train acc'f' {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')


# 定义train_with_data_aug函数，使用图像增广来训练模型
batch_size, devices, net = 256, try_all_gpus(), resnet18(10, 3)


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    # Adam优化器算是一个比较平滑的SGD，它对学习率调参不是很敏感
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_with_data_aug(train_augs, test_augs, net)