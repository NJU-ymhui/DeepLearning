import torch
from torch import nn
from d2l import torch as d2l
"""
使用正则化技术缓解过拟合
模型具有200维，使用只包含20个样本的小样本
"""


def init_params(number_features):
    """
    随机初始化模型参数
    :return:
    """
    # 随机化权重向量
    w = torch.normal(0, 1, (number_features, 1), requires_grad=True)
    # 随机化偏移量, 是一个形状为(1,)的零张量
    b = torch.zeros(1, requires_grad=True)
    # print(b.shape)
    return [w, b]


def l2_penalty(w):
    """
    定义L2范数惩罚
    :param w: 权重向量
    :return:
    """
    return w.pow(2).sum() / 2


def train(num_features, train_iter, test_iter, batch_size, regular=0):
    """
    :param regular: 正则系数
    :return: 拟合后的权重与偏移
    """
    num_epochs, lr = 100, 0.03  # 训练轮数, 学习率
    # 初始化模型参数
    w, b = init_params(num_features)
    # 选择模型，损失函数，优化器，学习率
    net = lambda x: d2l.linreg(x, w, b)  # 定义一个匿名函数，需要参数x
    loss = d2l.squared_loss  # 平方损失
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs],
                            legend=['train', 'test'])

    # 开始训练
    for epoch in range(num_epochs):
        for x, y in train_iter:  # train_iter的结构: (feature, label)
            # 选择性添加L2惩罚项
            l = loss(net(x), y) + regular * l2_penalty(w)  # net(x) = predict, loss(predict, label)为损失, 即loss(net(x), y)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size=batch_size)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('weight:')
    print(w.data[:5].numpy())
    print('bias:')
    print(b.data.numpy())
    print('L2:')
    print(torch.norm(w).item())
    return w.data, b.data


if __name__ == '__main__':
    n_train, n_test = 20, 100
    num_inputs = 200  # 200维(200个变量x)
    batch_size = 5
    # 函数真实的权重和偏移
    true_w = torch.ones((num_inputs, 1)) * 0.01
    true_b = 0.05
    print('true_w:')
    print(true_w[:10])
    print('true_b:')
    print(true_b)
    # 先得到数据，再生成迭代器
    # synthetic_data函数以N(0, 0.01^2)的高斯噪声为背景噪声，生成数据时自动添加
    train_data = d2l.synthetic_data(true_w, true_b, n_train)  # synthetic_data函数生成数据, 传入权重，偏移和生成数量
    train_iter = d2l.load_array(train_data, batch_size=batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size=batch_size, is_train=False)

    # 训练，分为开正则化和不开正则化
    # 先不开正则化
    print('----------no regularization----------')
    pred_w, pred_b = train(num_inputs, train_iter, test_iter, batch_size, regular=0)
    print('pred_w - true_w:')
    print(pred_w[:5] - true_w[:5])
    print('pred_b - true_b:')
    print(pred_b - true_b)
    d2l.plt.show()  # 可视化

    # 再开正则化
    print('----------with regularization----------')
    pred_w, pred_b = train(num_inputs, train_iter, test_iter, batch_size, regular=3)
    print('pred_w - true_w:')
    print(pred_w[:5] - true_w[:5])
    print('pred_b - true_b:')
    print(pred_b - true_b)
    d2l.plt.show()
