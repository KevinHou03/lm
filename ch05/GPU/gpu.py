
import torch
from torch import nn

'''计算设备'''
torch.device('cpu') # 创建设备对象
torch.cuda.device('cuda'), torch.cuda.device('cuda:0')

'''查询可用的gpu数量'''
print(torch.cuda.device_count()) # 0

'''如何在gpu不存在时运行代码'''
def try_gpu(i = 0):
    '''如果存在 返回gpu【i】否分返回cpu'''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    '''返回左右可用的gpu，否则cpu'''
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu())
print(try_gpu(2))
print(try_all_gpus())


'''查询tensor所在的设备'''

x = torch.tensor([1,2,3])
print(x.device) # cpu,默认在cpu上面

# 但是在创建的时候可以选择放在gpu上
x = torch.ones(2, 3, device=try_gpu())
y = torch.rand(2, 3, device = try_gpu(1))

'''如果要在gpu上运算一个a + b那么要保证ab都在同一个gpu上'''

'''在gpu上运行gpu'''
X = torch.ones(2, 3, device=try_gpu(1))
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device = try_gpu(2))
output = net(X) # 结果也会是在这个gpu上
