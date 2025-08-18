'''
微调（fine-tuning）指的是在已有的预训练模型基础上，使用目标任务/领域的少量数据继续训练，使模型参数从通用能力适配到特定场景；
与从零训练不同，微调通常用较小学习率、较短训练、并可只更新部分参数（如只训练分类头、解冻后几层，或用 LoRA/Adapter 等参数高效方法），
以在保留原有知识的同时降低过拟合与算力开销。它本质上是迁移学习的一种具体做法：把大规模预训练学到的表示迁移过来，再“顺一顺”到你的任务上；
需要注意，微调≠调参（hyperparameter tuning），后者是优化学习率、批大小等训练超参数，而微调是继续更新模型权重以完成领域适配。

关键点：
1. 更小的学习率，使用更少的数据迭代
2. 低层次的特征更通用，高层次则更与数据集本身相关
3. 可以固定一些底部的层的参数，不参与更新，以及更强的正则化

简而言之， 微调通过使用在大数据上得到的预训练好的模型来初始化新模型权重来提升精度
'''

import os
import torch
import torchvision
from torch import nn
import kagglehub
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from LM.ch05.GPU.gpu import try_all_gpus
from LM.utils import show_images, Accumulator, evaluate_accracy_gpu
from torchvision.models import resnet18, ResNet18_Weights

# Download latest version of hotdog dataset
path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")
print("Path to dataset files:", path)

train_imgs = ImageFolder(os.path.join(path, "train"))
test_imgs  = ImageFolder(os.path.join(path, "test"))

train_loader = DataLoader(train_imgs, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_imgs,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ImageFolder 的样本顺序是按类别文件夹名的字典序排的，且每个样本是个二元组 (image, label),所以label0在前面，label1在后面
hot_dogs = [train_imgs[i][0] for i in range(8)]
not_hot_dogs = [train_imgs[-i - 1][0] for i in range(8)]
# show_images(hot_dogs + not_hot_dogs, 2, 8, scale=1.4)

#数据增广， CenterCrop 的作用是从图像中心裁出固定大小的一块。
normalize = torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # 按该均值、方差做归一化
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])
# pretrained_net = torchvision.models.resnet18(pretrained=True) # pretrain = true意味着训练好的parameter也要拿过来
# print(pretrained_net.fc)

finetuned_net = resnet18(weights=ResNet18_Weights.DEFAULT)
finetuned_net.fc = nn.Linear(finetuned_net.fc.in_features, 2) # 你这个hotdog是一个两分类问题，所以你要把标号改成2
nn.init.xavier_uniform_(finetuned_net.fc.weight)



def train_ch130(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, devices=None,
                amp=False, grad_clip=None):
    """
    - amp: 是否使用混合精度only cuda可用时
    - grad_clip: 梯度裁剪的范数上限（如 1.0),None表示不裁剪
    """
    devices = devices or try_all_gpus()
    main_dev = devices[0]
    # 仅在有cuda的情况下启用Data Parallel
    if torch.cuda.is_available() and any(d.type == 'cuda' for d in devices):
        device_ids = [d.index for d in devices if d.type == 'cuda']
        net = nn.DataParallel(net, device_ids=device_ids).to(main_dev)
    else:
        net = net.to(main_dev)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and torch.cuda.is_available()))

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(4)#[loss_sum, correct_sum, n_for_loss, n_for_acc]

        for X, y in train_iter:
            X = X.to(main_dev, non_blocking=True)
            y = y.to(main_dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = net(X)
                    l = loss_fn(logits, y)
                    loss_mean = l.mean() if l.ndim > 0 else l
                scaler.scale(loss_mean).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = net(X)
                l = loss_fn(logits, y)
                loss_mean = l.mean() if l.ndim > 0 else l
                loss_mean.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

            bs = y.size(0)
            metric.add(loss_mean.item() * bs,
                       (logits.argmax(1) == y).sum().item(),
                       bs, y.numel())
        # evaluate 函数不接受 device 参数，就把 device=main_dev 去掉
        test_acc = evaluate_accracy_gpu(net, test_iter, device=main_dev)
        print(f'epoch {epoch + 1:02d} | '
              f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')

    return net

def train_fine_tuning(net, learning_rate, batch_size = 128, num_epoch = 5, param_group = True,
                      dry_run=True, dry_batches=5):
    print("fine tuning")
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(path,'train'), transform = train_augs),
        batch_size = batch_size, shuffle = True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(path,'test'), transform = test_augs),
        batch_size = batch_size, num_workers=0) # test no need shuffle
    devices = try_all_gpus()
    main_dev = devices[0]
    loss = nn.CrossEntropyLoss(reduction='none')

    if dry_run:
        print("dry-running")
        net.train()
        if torch.cuda.is_available():
            device_ids = [d.index for d in devices if d.type == 'cuda']
            net = nn.DataParallel(net, device_ids=device_ids).to(main_dev)
        else:
            net = net.to(main_dev)

        for b, (X, y) in enumerate(train_iter):
            if b >= dry_batches: break
            X = X.to(main_dev, non_blocking=True)
            y = y.to(main_dev, non_blocking=True)

            logits = net(X)
            l = loss(logits, y)
            loss_mean = l.mean()

            net.zero_grad(set_to_none=True)
            l.sum().backward()

            grad_sum, grad_l2 = 0.0, 0.0
            with torch.no_grad():
                for p in net.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        grad_sum += g.abs().sum().item()
                        grad_l2  += (g*g).sum().item()
            grad_l2 = grad_l2 ** 0.5
            acc = (logits.argmax(1) == y).float().mean().item()

            print(f"[dry_run] batch {b+1}/{dry_batches} "
                  f"loss={loss_mean.item():.4f} acc={acc:.3f} "
                  f"grad_sum={grad_sum:.1e} grad_l2={grad_l2:.1e} "
                  f"logits_shape={tuple(logits.shape)}")

        # 不进入正式训练，直接返回
        return

    if param_group:
        # 除了最后一层的learning rate外，用的是默认的learning rate
        # 最后一层的learning rate用的是十倍的learning rate
        params_lx = [
            param for name, param in net.named_parameters()
            if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([
            {'params': params_lx},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    train_ch130(net, train_iter, test_iter, loss, trainer, num_epoch, devices)

if __name__ == '__main__':
    print('running')
    train_fine_tuning(finetuned_net, 5e-5)