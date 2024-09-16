import torch
from torch import nn
from d2l import torch as d2l
"""利用深度学习框架高级api实现暂退法，请先阅读dropout_self.py"""


def init_weights(model):
    """
    初始化权重，针对此例
    :param model: 传入模型
    :return:
    """
    if type(model) == nn.Linear:
        nn.init.normal_(model.weight, 0, 0.01)  # 等价于 nn.init.normal_(model.weight, std=0.01) mean默认0，std默认1.


if __name__ == '__main__':
    # 初始化二层感知机的参数，使用Fashion-MNIST数据集
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    drop_out1, drop_out2 = 0.2, 0.5  # 将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5

    # 定义模型
    net = nn.Sequential(
        # 该函数(nn.Flatten())将多维输入张量展平为一维，常用于神经网络中连接卷积层与全连接层
        nn.Flatten(),  # 将输入展平
        # 第一层隐藏层
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_out1),
        # 第二层隐藏层
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(drop_out2),
        # 输出层
        nn.Linear(num_hiddens2, num_outputs)
    )

    net.apply(init_weights)  # 初始化权重

    # 训练和测试
    num_epochs, lr, batch_size = 10, 0.5, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()
