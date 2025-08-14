import torch
from torch import nn
from LM.ch02.Softmax.MNIST_DS import load_data_fashion_mnist
from LM.ch02.Softmax.softmax_scratch import train_ch3
from dropout import dropout_layer

# 设置dropout
dropout1, dropout2 = 0.2, 0.5
# 定义具有两个隐藏层的mlp， 没个隐藏层有256个单元
class Net(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training

        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)

        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))

        if self.training: # dropout只在训练时使用
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))

        if self.training:
            H2 = dropout_layer(H2, dropout2)
        output = self.lin3(H2)

        return output


# api简洁实现

net_w_dropoutAPI = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(),
    nn.Dropout(dropout2), nn.Linear(256, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)

net_w_dropoutAPI.apply(init_weights)



if __name__ == "__main__":
    # model params
    num_inputs, num_outputs, num_hiddens1, num_hidden2 = 784, 10, 256, 256
    # training params
    num_epochs, lr, batch_size = 1, 0.5, 10
    # the model
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hidden2, True)
    #net = net_w_dropoutAPI
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)




