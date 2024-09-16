import torch
from torch import nn
from d2l import torch as d2l
"""从零实现暂退法"""


def dropout_layer(X, drop_prob):
    """
    在隐藏层应用暂退法，用于神经网络训练中防止过拟合
    该函数以dropout的概率丢弃张量输入X中的元素，重新缩放剩余部分即除以 1 - dropout
    :param X: 张量输入
    :param drop_prob: 概率
    :return: 丢弃、放缩后的结果
    """
    assert 0 <= drop_prob <= 1
    if drop_prob == 1:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > drop_prob).float()  # 生成一个形状与X相同、元素大于drop_prob的随机数掩码mask, 不大于的位置为0
    return mask * X / (1 - drop_prob)


# 定义模型
# 为每一层分别设置暂退概率
drop_out1, drop_out2 = 0.2, 0.5  # 将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5


class Net(nn.Module):
    """实现一个两层感知机"""
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)  # 第一个隐藏层
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)  # 第二个隐藏层
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)  # 输出层
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 训练模型时启用dropout
        if self.training:
            H1 = dropout_layer(H1, drop_out1)
        H2 = self.relu(self.lin2(H1))
        # 同理，训练模型时启用dropout, 防止测试时也启用dropout
        if self.training:
            H2 = dropout_layer(H2, drop_out2)
        output = self.lin3(H2)
        return output


if __name__ == '__main__':
    # 测试暂退函数
    A = torch.arange(25).reshape(5, 5)
    print('before:')
    print(A)
    print('after:')
    print(dropout_layer(A, 0))
    print(dropout_layer(A, 0.5))
    print(dropout_layer(A, 1))

    # 定义模型参数，依然使用Fashion-MNIST数据集
    # 输入层28*28个神经元，输出层10个神经元, 有两个隐藏层，每个隐藏层有256个神经元
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    # 定义模型
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    # 训练和测试
    num_epochs, lr, batch_size = 10, 0.5, 256  # 迭代轮数， 学习率， 批量大小
    loss = nn.CrossEntropyLoss(reduction='none')  # 损失函数
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 获取训练迭代器，测试迭代器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
    # 传参顺序为：模型，训练集，测试集，损失函数，迭代轮数，优化器
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()
