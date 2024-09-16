import math
import torch
import random
import numpy as np
from torch import nn
from d2l import torch as d2l


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上的模型损失"""
    metric = d2l.Accumulator(2)  # 损失的综合，样本数量
    for X, y in data_iter:
        # 将输入数据X通过神经网络net进行前向传播，得到输出结果out
        out = net(X)
        # 将标签y的形状重塑为模型输出out的形状，确保维度匹配
        y = y.reshape(out.shape)
        # 计算模型输出out与标签y之间的损失函数值，并求和
        l = loss(out, y)
        # 更新评估指标，累加损失总和及样本数量
        metric.add(l.sum(), y.numel())

    # 计算并返回两个metric元素的除法结果
    # 此函数解释两个metric元素之间的比例关系，其中metric假设为一个包含两个元素的列表或元组
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    """
    :param train_features:
    :param test_features:
    :param train_labels:
    :param test_labels:
    :param num_epochs:
    :return:
    """
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 多项式中已有偏置，所以不必再设置
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    # 设置批量大小，训练迭代器，测试迭代器，训练器
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 初始化一个动画对象animation，用于绘制训练和测试损失曲线
    animation = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                             legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)  # 对每个epoch，调用d2l.train_epoch_ch3训练模型
        # 方便可视化
        if epoch == 0 or (epoch + 1) % 20 == 0:
            # 若为首个epoch或当前epoch加1能被20整除，记录训练和测试损失并添加到动画中
            animation.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))

    # 输出权重
    print("weight:")
    print(net[0].weight.data.numpy())
    return net[0].weight.data


if __name__ == '__main__':
    # 拟合时假定阶数为20(其实是19，还有0次项)
    max_degree = 20
    # 生成一个三阶多项式
    n_train, n_test = 100, 100
    true_w = torch.zeros(max_degree)
    true_w[0:4] = torch.tensor([5, 1.2, -3.4, 5.6])  # 多项式的系数 x^0 x^1 x^2 x^3
    # print(true_w)

    # 生成初始数据x
    features = torch.randn((n_train + n_test, 1))
    # 随机打乱
    random_indices = torch.randperm(n_train + n_test)
    features = features[random_indices]

    # print(features)
    # 生成x的幂次
    poly_features = torch.pow(features, torch.arange(max_degree).reshape(1, -1))  # 构造多项式特征, 幂次从0开始
    # print(poly_features)
    # 消减梯度
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(i+1) = (i+1)!, 防止梯度增加过快

    # 计算标签值，y = w_0 + w_1 * x + w_2 * x^2 + w_3 * x^3 + ... = w · x
    # 一共有多个样本，所以x的幂次样本poly_features是一个矩阵
    labels = torch.mm(poly_features, true_w.reshape(-1, 1))  # true_w是一个行向量，所以用.reshape(-1,1)变成列向量
    # 加上噪声
    labels += torch.normal(0, 0.1, labels.shape)

    # 看一眼数据
    print("data slices:")
    print(features[:2])
    print(poly_features[:2, :])
    print(labels[:2])

    # 训练
    # 数据集前面是验证集，后面是训练集

    # 先看正常拟合
    # 取前四个特征，即w_0 + w_1 * x + w_2 * x^2 + w_3 * x^3，正好是目标的阶
    predict_w = train(poly_features[:n_train, :4], poly_features[n_test:, :4], labels[:n_train], labels[n_test:])
    print('correct mistake:')
    print(predict_w - true_w[:4])

    # 欠拟合
    # 因为实际是一个三级多项式，当我们尝试用线性模型（即一次函数）去拟合时，会出问题
    # 只取特征的前两行，即w_0 + w_1 * x
    predict_w = train(poly_features[:n_train, :2], poly_features[n_test:, :2], labels[:n_train], labels[n_test:])
    print('linear mistake:')
    print(predict_w - true_w[:2])

    # 过拟合
    # 当模型过于复杂时可能发生过拟合，比如我们取前8个特征
    # 即w_0 + w_1 * x + w_2 * x^2 + w_3 * x^3 + w_4 * x^4 + w_5 * x^5 + w_6 * x^6 + w_7 * x^7七阶多项式
    predict_w = train(poly_features[:n_train, :8], poly_features[n_test:, :8], labels[:n_train], labels[n_test:])
    print('overfit mistake:')
    print(predict_w - true_w[:8])

    # 取所有特征
    predict_w = train(poly_features[:n_train, :], poly_features[n_test:, :], labels[:n_train], labels[n_test:])
    print('all mistake:')
    print(predict_w - true_w)
    d2l.plt.show()  # 可视化
