import torch
import torchvision
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


trans = transforms.ToTensor()

mnist_train = datasets.FashionMNIST(
    root='/Users/kevinhou/Downloads/pythonProject/LM/datasets',
    train=True,
    transform=trans,
    download=False
)

mnist_test = datasets.FashionMNIST(
    root='/Users/kevinhou/Downloads/pythonProject/LM/datasets',
    train=False,
    transform=trans,
    download=False
)


def get_fashion_mnist_labels(labels):
    '''返回数据集的文本标签'''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int (i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    '''Plot a list of images'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()


    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)

def get_dataloader_worker():
    return 4

def load_data_fashion_mnist(batch_size, resize = None):
    '''下载这个数据集，并且把它加载到内存中'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = '/Users/kevinhou/Downloads/pythonProject/LM/datasets', train = True, transform = trans, download=False) # 我将手动提取
    mnist_test = torchvision.datasets.FashionMNIST(root='/Users/kevinhou/Downloads/pythonProject/LM/datasets', train=False,transform=trans, download=False)

    return data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = get_dataloader_worker()), data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = get_dataloader_worker())

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18,28,28), 2, 9)

