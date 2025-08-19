import os
import random
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34
from torchvision.models.quantization import resnet50

from LM.ch05.GPU.gpu import try_all_gpus
from LM.utils import Accumulator, evaluate_accracy_gpu, accuracy# 假定可用

# Config
DATA_DIR = "./data"
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 5
VALID_RATIO = 0.1
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 16
BATCH_SIZE_TEST  = 16
NUM_WORKERS = 0
SEED = 42


def get_tiny_cifar10(
    root: str = DATA_DIR,
    train_per_class: int = TRAIN_PER_CLASS,
    test_per_class: int = TEST_PER_CLASS,
    seed: int = SEED
) -> Tuple[Subset, Subset, List[str]]:
    random.seed(seed)

    tfm_train = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_full = datasets.CIFAR10(root=root, train=True,  download=True, transform=tfm_train)
    test_full  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_test)

    train_indices = []
    for cls in range(10):
        idx = [i for i, y in enumerate(train_full.targets) if y == cls]
        train_indices.extend(idx[:train_per_class])
    tiny_train = Subset(train_full, train_indices)

    test_indices = []
    for cls in range(10):
        idx = [i for i, y in enumerate(test_full.targets) if y == cls]
        random.shuffle(idx)
        test_indices.extend(idx[:test_per_class])
    tiny_test = Subset(test_full, test_indices)

    return tiny_train, tiny_test, train_full.classes

def split_train_valid(
    tiny_train: Subset,
    valid_ratio: float = VALID_RATIO,
    seed: int = SEED
) -> Tuple[Subset, Subset]:
    n_total = len(tiny_train)
    n_valid = max(1, int(round(n_total * valid_ratio)))
    n_train = n_total - n_valid
    generator = torch.Generator().manual_seed(seed)
    train_subset, valid_subset = random_split(tiny_train, [n_train, n_valid], generator=generator)
    return train_subset, valid_subset


def get_loaders(
    train_subset: Subset,
    valid_subset: Subset,
    test_subset: Subset
):
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE_VALID, shuffle=False,num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE_TEST,  shuffle=False,num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return train_loader, valid_loader, test_loader

def get_net():
    # 输入通道默认 3；使用关键字参数设置类别数
    return resnet34(weights=None, num_classes=10)


loss_fn = nn.CrossEntropyLoss(reduction="none")

def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])

    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l_sum = l.sum()
    l_sum.backward()
    trainer.step()

    correct = accuracy(pred, y)
    if torch.is_tensor(correct):
        correct = correct.item()

    return l_sum.item(), correct#返回scalar value


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus()):
    if devices is None:
        devices = try_all_gpus()
    num_batches = len(train_iter)
    # nn.DataParallel使用多GPU
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch_ch13(net,features,labels,loss,trainer,devices)
            metric.add(l,acc,labels.shape[0],labels.numel())
        test_acc = evaluate_accracy_gpu(net,test_iter)
        print(f'epoch {epoch + 1:02d} | '
              f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')


def train(
    net: nn.Module,
    train_iter: DataLoader,
    valid_iter: DataLoader,
    num_epochs: int,
    lr: float,
    wd: float,
    lr_period: int,
    lr_decay: float,
    device: torch.device
):
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_period, gamma=lr_decay)

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)  # sum_loss, sum_correct, sum_samples
        for features, labels in train_iter:
            l, acc = train_batch_ch13(net, features, labels, loss_fn, optimizer, [device])
            metric.add(l, acc, labels.shape[0])

        if valid_iter is not None:
            valid_acc = evaluate_accracy_gpu(net, valid_iter)

        scheduler.step()

        train_loss = metric[0] / metric[2]
        train_acc  = metric[1] / metric[2]
        if valid_iter is not None:
            print(f"epoch {epoch+1}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, valid acc {valid_acc:.3f}")
        else:
            print(f"epoch {epoch+1}: train loss {train_loss:.3f}, train acc {train_acc:.3f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiny_train, tiny_test, classes = get_tiny_cifar10(DATA_DIR, TRAIN_PER_CLASS, TEST_PER_CLASS, SEED)
    train_subset, valid_subset = split_train_valid(tiny_train, VALID_RATIO, SEED)
    train_loader, valid_loader, test_loader = get_loaders(train_subset, valid_subset, tiny_test)

    print("Classes:", classes)
    print(f"Train/Valid/Test sizes: {len(train_subset)} / {len(valid_subset)} / {len(tiny_test)}")

    # 快速 sanity check
    xb, yb = next(iter(train_loader))
    print("Batch shapes:", xb.shape, yb.shape)

    net = get_net()
    train(
        net=net,
        train_iter=train_loader,
        valid_iter=valid_loader,
        num_epochs=40,
        lr=2e-4,
        wd=5e-4,
        lr_period=4,
        lr_decay=0.9,
        device=device
    )
