import random
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
"""
从零开始实现一个线性回归模型
"""


def emit_data(w, b, num_examples):
    """
    create dataset with noise
    :w: params of weight
    :b: offset
    :num_examples: number of examples
    :return:
    """
    # according to y = Xw + b + epsilon, X is dataset, w is params, b is offset, epsilon is noise
    shape = (num_examples, len(w))  # shape of dataset, examples.number * w.length (Xw need X.col == w.row)
    X = torch.normal(0, 1, shape)  # dataset X from standard normal distribution
    y = torch.matmul(X, w) + b
    # add noise epsilon
    noise = torch.normal(0, 0.01, y.shape)
    y += noise  # add noise
    return X, y.reshape((-1, 1))


def data_iter(batch_size, x, y):
    """
    读取数据集，根据原数据集可以生成一些批量样本
    :param batch_size:批量样本的大小
    :param x:
    :param y:
    :return:
    """
    number_examples = len(x)  # x is dataset features, length is number of examples
    indices = list(range(number_examples))
    random.shuffle(indices)  # “洗牌”，对列表进行随机排序，使列表中的每个元素都有可能出现在任意位置
    for i in range(0, number_examples, batch_size):  # 以指定的批量大小为步长，遍历列表，每次生成一个批量样本
        batch_indices = torch.tensor(indices[i:min(i + batch_size, number_examples)])  # 剩下不足一个批量大小的数据生成到一起
        yield x[batch_indices], y[batch_indices]


def linear_model(X, w, b):
    """
    define a linear model
    :return:
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    定义均方损失函数
    :param y_hat: predict
    :param y: actual
    :return:
    """
    # 在此例中我们有两个权重（特征）
    # sum((y_pred - y_act) ^ 2) / 2
    return (y_hat - y) ** 2 * 0.5


def sgd(param, lr, batch_size):
    """
    定义优化算法：梯度下降
    :param param 模型参数列表
    :param lr 学习率，控制参数更新的幅度
    :param batch_size 批量大小，用于计算梯度平均值
    :return:
    """
    with torch.no_grad():  # 禁用自动梯度计算，因为参数更新不需要跟踪梯度
        for param_i in param:
            param_i.data -= lr * param_i.grad / batch_size  # param_i.grad 是当前参数的梯度，除以 batch_size 以得到平均梯度
            param_i.grad.zero_()  # 梯度清零，为下一次迭代计算新的梯度做准备


def func(k, b, x):
    return k * x + b


if __name__ == '__main__':
    # actual params and offset
    actual_w = torch.tensor([1.14, 5.14])
    actual_b = 1.91981
    data_size = 1145

    # emit data
    features, labels = emit_data(actual_w, actual_b, data_size)
    print("features:")
    print(features)
    print("labels:")
    print(labels)
    # data in vision
    d2l.set_figsize()  # 设置绘图的尺寸
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)  # 绘制散点图
    # features[:, 1]表示特征的第二列数据, labels表示标签；.detach().numpy()用于从张量中提取数值。
    plt.xlabel = "features"
    plt.ylabel = "labels"
    # plt.show()

    # read data
    batch_size = 10  # 批量大小为10
    print("a batch of data:")
    for X, y in data_iter(batch_size, features, labels):
        print(X)
        print(y)
        break  # 看一下就行了

    # init model's param
    # 创建了一个名为w的张量，形状为(2, 1)，值从均值为0、标准差为0.01的正态分布中随机生成，且要求计算梯度。
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # 作为模型初始权重
    # 创建了一个名为b的张量，形状为(1,)，初始值为0，同样要求计算梯度。这些操作通常用于初始化神经网络的权重和偏置项。
    b = torch.zeros(1, requires_grad=True)  # 作为模型初始偏移
    # 初始化这些参数之后，我们需要不断更新它们，直到足够拟合数据

    # start training
    # 首先为模板赋上此例中的具体值
    lr = 0.03  # 学习率是一个超参数，暂时指定为0.03
    num_epochs = 3  # 训练轮数也是一个超参数
    net = linear_model  # 神经网络采用线性模型
    loss = squared_loss  # 损失函数采用均方损失函数
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 计算X和y的小批量损失
            l.sum().backward()  # 反向传播，计算梯度
            sgd([w, b], lr, batch_size)  # 更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    print("real:")
    print(actual_w)
    print(actual_b)
    print("predict:")
    print(w)
    print(b)
    # in vision, real scatter is blue, predict linear is red
    draw_x = features[:, 1]
    draw_y = func(w[1], b, draw_x)
    d2l.plt.plot(draw_x, draw_y.detach().numpy(), color="red")
    plt.show()
