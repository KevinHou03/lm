import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

train_data = pd.read_csv('/Users/kevinhou/Documents/PyTorch_Datasets/kaggle_house_price/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/Users/kevinhou/Documents/PyTorch_Datasets/kaggle_house_price/house-prices-advanced-regression-techniques/test.csv')
print(train_data.shape, test_data.shape) # (1460, 81) (1459, 80)
print(train_data.iloc[0:5, [0,1,2,3,-3,-2,-1]])

# 第一列是id，我们不需要，所以拿掉
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features)
# 将所有缺失value替换为平均值，然后标准化所有数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 这里的all_features.dtypes返回一个series -> feat: dtype
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 用one-hot替换离散值
all_features = pd.get_dummies(all_features, dummy_na=True) # get_dummies把离散变量转化为one-hot编码
print(all_features.shape)
print(all_features.dtypes.value_counts()) # 检查之后发现还是有bool类型，所以全部转化为0/1
all_features = all_features.astype(np.float32)


# 转化为tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32) #.values用来讲dataframe/series转化酶numpy
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 训练
loss = nn.MSELoss()
in_features = train_features.shape[1]

def load_array(data_arrays, batch_size, is_train = True):
    # 构造一个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size = batch_size, shuffle = is_train)

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return net

# 对于房价来说，有一千万和十万，那么造成的误差也会更大，所以我们用log来处理误差 -> 相对误差
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf')) # min值为1是因为预测结果有可能为0或者负数，log会报错
    '''clamp为限制范围函数，将输入张量的每个元素限制在指定区间内，低于最小值/高于最大值都会被取最小值/最大值'''
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels))) # 检查均方log误差公式
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epoch, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def train_and_pred(train_features, test_features, train_labels, test_labels, num_epoch, learning_rate, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epoch, learning_rate, weight_decay, batch_size)
    print(f'train log rmse {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    # test_data['SalePrice'] = pd.Series(preds.reshape(-1, 1)[0])
    test_data['SalePrice'] = preds.reshape(-1)
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

# k-fold验证： 把整个训练集划分为K份，每次用K-1份做训练，剩下1份做验证，重复K次
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) #
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epoch, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epoch, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(f'fold{i + 1}, train_log_rmse {float(train_ls[-1])}, valid_log_rmse {float(valid_ls[-1])}')
    return train_l_sum / k, valid_l_sum / k



k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k} fold validation: avg log rmse: {float(train_l):f}, avg valid log rmse: {float(valid_l):f}')

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)