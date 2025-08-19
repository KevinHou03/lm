from torchvision import datasets, transforms
from torch.utils.data import Subset
import random


def get_tiny_cifar10(root = '/data'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 共10类，取每类前1000张训练图像
    train_indices = []
    for cls in range(10):
        cls_idx = [i for i, y in enumerate(trainset.targets) if y == cls]
        train_indices.extend(cls_idx[:1000])

    tiny_trainset = Subset(trainset, train_indices)

    # 随机取每一类五张测试图像
    test_indices = []
    for cls in range(10):
        cls_idx = [i for i, y in enumerate(testset.targets) if y == cls]
        test_indices.extend(random.sample(cls_idx, 5))

    #生成一个mini dataset
    tiny_testset = Subset(testset, test_indices)

    print(f"Tiny Train size: {len(tiny_trainset)}")# 10000
    print(f"Tiny Test size: {len(tiny_testset)}") # 50

    return tiny_trainset, tiny_testset

train_set, test_set = get_tiny_cifar10()


import csv
def save_cifar10_labels(dataset, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for idx, (img, label) in enumerate(dataset):
            writer.writerow([f"{idx:05d}.png", dataset.dataset.classes[label]])

save_cifar10_labels(train_set, 'data/cifar10_labels.csv')
