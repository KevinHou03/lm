'''给你参数params，全部移到device：GPU上'''
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()# 把这个张量标记为需要梯度，优化器才有东西可更新
    return new_params

'''
在所有的gpu上有一些data，把他们放到同一个gpu上加起来（此操作只能在同一个gpu上使用），
把一个张量列表 data（可能分布在不同设备上）的值相加到 data[0]，然后把结果拷回每个元素，
实现“所有副本都拿到和”的简易 all-reduce。
'''
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device) # 把结果赋给其他位置





