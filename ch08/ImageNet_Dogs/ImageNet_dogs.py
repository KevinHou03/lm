import os
import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet34_Weights, resnet34

from LM.ch05.GPU.gpu import try_all_gpus
from LM.utils import Accumulator

demo = True

IMAGES_ROOT = "/Users/kevinhou/Downloads/pythonProject/LM/ch08/ImageNet_Dogs/data/images/Images"
batch_size = 16 if demo else 128
valid_ratio = 0.1


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0, 4.0/3.0)),  # 注意这里和之前不同，图片比例3：4到4：3
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),  # 0.4代表上下取40%
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                    [0.229,0.224,0.225])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])


full_ds = datasets.ImageFolder(IMAGES_ROOT, transform=transform_train)
n_val = int(len(full_ds) * valid_ratio)
train_ds, val_ds = random_split(
    full_ds, [len(full_ds) - n_val, n_val],
    torch.Generator().manual_seed(42)
)
# 验证集用验证的增广
val_ds.dataset.transform = transform_val
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

print("num classes:", len(full_ds.classes))
print("train/val:", len(train_ds), len(val_ds))


def get_net(device, num_classes=120, freeze_backbone=True):
    #1. 载入预训练
    weights = ResNet34_Weights.DEFAULT
    net = resnet34(weights=weights)
    #2. 是否backbone（除最后分类层外的特征提取部分）
    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False # freeze -> no update
    #3. 替换分类头（原fc.in_features=512）
    in_f = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return net.to(device)

# loss
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    net.eval()
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices), labels.to(devices)
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum().item()
        n += labels.numel()
    return l_sum / n


# 训练函数
# def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
#     if len(devices) > 1 and devices[0].type == 'cuda':
#         net = nn.DataParallel(net, device_ids=devices).to(devices[0])
#     else:
#         net = net.to(devices[0])
#     trainer = torch.optim.SGD(
#         (param for param in net.parameters() if param.requires_grad),#只更新某一部分
#         lr=lr, momentum=0.9, weight_decay=wd)
#     scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay) # 学习率更新
#     num_batches  = len(train_iter)
#     legend = ['train loss']
#     if valid_iter is not None:
#         legend.append('valid loss')
#
#     for epoch in range(num_epochs):
#         net.train()
#         metric = Accumulator(2)
#         for i, (features, labels) in enumerate(train_iter):
#             features, labels = features.to(devices[0]), labels.to(devices[0])
#             trainer.zero_grad()
#             output = net(features)
#             l = loss(output, labels).sum()
#             l.backward()
#             trainer.step()
#             metric.add(l.item(), labels.shape[0])
#         measures = f'train loss {metric[0] / metric[1]:.3f}'
#         if num_batches >= 5 and ((i + 1) % (num_batches // 5) == 0 or i == num_batches - 1):
#             pass
#         train_loss = metric[0] / metric[1]
#         if valid_iter is not None:
#             valid_loss = evaluate_loss(valid_iter, net, devices[0])
#             print(f"epoch {epoch+1:02d} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f}")
#         else:
#             print(f"epoch {epoch+1:02d} | train loss {train_loss:.4f}")
#         scheduler.step()

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay, max_steps=None):
    if len(devices) > 1 and devices[0].type == 'cuda':
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else:
        net = net.to(devices[0])

    opt = torch.optim.SGD((p for p in net.parameters() if p.requires_grad),
                          lr=lr, momentum=0.9, weight_decay=wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_period, gamma=lr_decay)

    seen = 0
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(2)  # [sum_loss, sum_samples]
        for X, y in train_iter:
            X, y = X.to(devices[0]), y.to(devices[0])
            opt.zero_grad()
            out = net(X)
            l = loss(out, y).sum()
            l.backward()
            opt.step()
            metric.add(l.item(), y.size(0))

            seen += 1
            if max_steps is not None and seen >= max_steps:
                train_loss = metric[0] / metric[1]
                if valid_iter is not None:
                    # 建议把你的 evaluate_loss 改为接收单个 device；这里示例传 devices[0]
                    valid_loss = evaluate_loss(valid_iter, net, devices[0])
                    print(f"[quick] epoch {epoch+1} step {seen} | train {train_loss:.4f} | valid {valid_loss:.4f}")
                else:
                    print(f"[quick] epoch {epoch+1} step {seen} | train {train_loss:.4f}")
                return
        sch.step()

if __name__ == '__main__':
    device = torch.device('cpu')
    devices = [device]
    num_epochs, lr, wd = 10, 1e-4, 1e-4
    lr_period, lr_decay = 2, 0.9

    net = get_net(device)
    train(net, train_loader, val_loader,
          num_epochs, lr, wd, devices, lr_period, lr_decay, max_steps=1)
