from d2l import torch as d2l
import torch
from torch import nn


def init_weights(m):
    """
    初始化神经网络模型中的权重
    :param m: 传入模块
    """
    # 检查传入模块是否为全连接
    if type(m) == nn.Linear:
        # 如果是，就以均值为0，标准差为0.01的正态分布对权重进行初始化
        nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':
    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 定义使用的模型
    net = nn.Sequential(
        nn.Flatten(),  # 将输入展平为向量
        nn.Linear(784, 256),  # 输入为28*28=784，隐藏层为256
        nn.ReLU(),  # 激活函数为ReLU
        nn.Linear(256, 10)  # 隐藏层为256，输出为10
    )

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 训练
    num_epochs, lr = 10, 0.1
    # 直接从pytorch的优化算法类获取优化器SGD，传入网络参数和学习率
    trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 用于训练神经网络net，设置学习率为lr，优化器将根据此学习率和反向传播计算的梯度来更新net的所有可训练参数
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
