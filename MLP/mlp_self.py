import torch
from matplotlib import pyplot as plt
from d2l import torch as d2l
from torch import nn
"""从零开始写一个多层感知机"""


# Fashion-MNIST图像数据集的输入是28*28的灰度图像，输出是10个类别
# 不妨实现一个256个隐藏层节点的MLP
number_inputs, number_outputs, number_hidden = 28 * 28, 10, 256


def relu(x):
    """relu激活函数"""
    zero = torch.zeros_like(x)
    return torch.max(x, zero)


if __name__ == '__main__':
    # 继续使用Fashion-MNIST图像数据集
    batch_size = 256  # 批量样本大小
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载数据集并划分为训练迭代器和测试迭代器

    # 初始化模型参数
    # 注意到我们的多层感知机一共需要两层的参数（参数是一个权重矩阵和一个偏移向量），两层分别是输入层到隐藏层和隐藏层到输出层
    # 初始化为小幅度随机数, shape=(num_inputs, num_hidden)，因为是input层到hidden层，所以形状是input * hidden
    w1 = nn.Parameter(torch.randn(number_inputs, number_hidden, requires_grad=True) * 0.01)
    # 初始化为0, 长度为num_hidden
    b1 = nn.Parameter(torch.zeros(number_hidden, requires_grad=True))
    # 初始化为小幅度随机数, shape=(num_hidden, num_outputs)，因为是hidden层到output层，所以形状是hidden * output
    w2 = nn.Parameter(torch.randn(number_hidden, number_outputs, requires_grad=True) * 0.01)
    # 初始化为0, 长度为num_outputs
    b2 = nn.Parameter(torch.zeros(number_outputs, requires_grad=True))
    # 参数列表为 w1, b1, w2, b2
    params = [w1, b1, w2, b2]

    def net(x):
        """定义模型"""
        x = x.reshape((-1, number_inputs))  # 将输入x重塑为二维数组，形状为(-1, number_inputs)，-1表示自动计算样本数量, 第二维大小为number_inputs
        hidden = relu(x @ w1 + b1)  # 计算隐藏层输出：使用ReLU激活函数对 输入与权重w1矩阵做乘法后加上偏置b1的结果 进行激活
        return hidden @ w2 + b2  # @表示矩阵乘法, 等价于torch.matmul(hidden, w2) + b2
    # 在线性回归一文从零实现softmax板块中已实现损失函数，此处直接调用现有库api
    loss = nn.CrossEntropyLoss(reduction='none')

    # 训练
    num_epochs, lr = 10, 0.1  # 迭代轮数，学习率
    updator = torch.optim.SGD(params, lr=lr)
    # 模型的训练与softmax一致，因此此处直接调用现有库api
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updator)  # 该函数用于训练模型，输入参数为网络模型、训练数据、测试数据、损失函数、迭代轮数、优化器
    d2l.plt.show()

    # 预测 / 检查预测结果
    d2l.predict_ch3(net, test_iter)
