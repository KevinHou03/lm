import torch
from torch import nn
from torch.nn import functional as F


'''1. 加载和保存tensor'''
x = torch.arange(4)
print(x) # tensor([0, 1, 2, 3])

torch.save(x, 'x-file') # save 为 x-file

x2 = torch.load('x-file')
print(x2) # tensor([0, 1, 2, 3])

'''2. 储存一个tensor列表，并且存取'''
y = torch.zeros(4)
torch.save([x, y], 'xy-list')
x2, y2 = torch.load('xy-list')
print(x2, y2) # tensor([0, 1, 2, 3]) tensor([0., 0., 0., 0.])

'''3. 写入或者读取从str映射到tensor的字典'''
mydict = {'x': x, 'y': y}
torch.save(mydict, 'xy-dict')
mydict2 = torch.load('xy-dict')
print(mydict2) # {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}


'''加载和保存模型参数'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size = (2, 20))
y = net(X)

torch.save(net.state_dict(), 'MLPnet.params')

'''如何load参数呢'''
clone = MLP()
clone.load_state_dict(torch.load('MLPnet.params')) # 加载之前训练好的参数权重
clone.eval()

y_clone = clone(X)
print(y_clone == y) # true

