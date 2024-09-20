import torch
from torch import nn
from d2l import torch as d2l
from visualization import show_heatmaps


def data_f(x):
    """生成训练数据"""
    return 2 * torch.sin(x) + x ** 0.8 + torch.normal(0, 0.5, x.shape)


def true_f(x):
    """生成真实数据"""
    return 2 * torch.sin(x) + x ** 0.8


if __name__ == '__main__':
    # 生成数据集
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 为了之后更好地可视化，将训练数据集排序
    y_train = data_f(x_train)
    x_test = torch.arange(0, 5, 0.1)
    y_test = true_f(x_test)
    n_test = len(x_test)
    print(n_test)

    def plot_kernel_reg(y_hat):
        """绘制样本，不带噪声项的真实函数记为Truth，预测出的函数记为Pred"""
        d2l.plot(x_test, [y_test, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
                 xlim=[0, 5], ylim=[-1, 5])
        d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
        d2l.plt.show()

    # 基于平均汇聚
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)

    # 基于非参数的注意力汇聚
    # 将x_test重复n_train次，然后重塑为矩阵，以准备进行注意力计算
    x_repeat = torch.repeat_interleave(x_test, n_train).reshape((-1, n_train))
    # 计算注意力权重，通过计算x_repeat和x_train之间的差异，应用softmax函数进行标准化
    attention_weights = nn.functional.softmax(-(x_repeat - x_train) ** 2 / 2, dim=1)
    # 通过加权训练数据的目标变量y_train来预测y_hat，权重为attention_weights
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)
    show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),  # unsqueeze添加两个维度
                  xlabel='sorted training inputs',
                  ylabel='sorted testing inputs')

    # 基于带参数的注意力汇聚
    class NWKernelReg(nn.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.w = nn.Parameter(torch.rand((1, ), requires_grad=True))  # 初始化一个可训练参数w

        def forward(self, queries, keys, values):
            """
            前向传播方法，实现自注意力机制的计算。
            :param queries: 查询张量，模型的输入之一。
            :param keys: 键张量，用于计算注意力权重。
            :param values: 值张量，用于加权计算最终输出。
            :return: 经过注意力机制计算后的输出张量。
            """
            # 将查询张量(queries)重复展开，以匹配键张量(keys)的维度
            queries = torch.repeat_interleave(queries, keys.shape[1]).reshape((-1, keys.shape[1]))
            # 计算注意力权重
            # 通过softmax函数对计算出的权重进行归一化，使其和为1
            self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
            # 应用注意力权重
            # 通过加权求和得到最终的输出
            return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

    # 训练

    # 数据处理
    # 将训练集中的每个样本沿着第一维度重复n_train次，生成新的训练数据
    # 这样做的目的是为了在模型训练过程中增加样本多样性，增强模型的泛化能力
    x_tile = x_train.repeat((n_train, 1))
    # 同样的操作应用于标签数据，保证每个重复的样本仍然保留其正确的标签
    # 这是为了在增加样本多样性的同时，保持样本与其标签的一一对应关系
    y_tile = y_train.repeat((n_train, 1))
    # 通过屏蔽掉对角线元素，选择键值对
    keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    # 模型选择，且在带参数的注意力汇聚中使用平方误差损失函数和随机梯度下降优化
    net = NWKernelReg()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        # 将模型参数的梯度归零
        net.zero_grad()
        # 计算当前周期的损失
        l = loss(net(x_train, keys, values), y_train)
        # 反向传播损失
        l.sum().backward()
        # 更新模型参数
        trainer.step()
        # 可视化
        print(f'epoch %d, loss: %f' % (epoch + 1, l.sum()))
        animator.add(epoch + 1, float(l.sum()))
    # 查看损失迭代情况
    d2l.plt.show()

    # 测试
    keys = x_train.repeat((n_test, 1))  # 注意要用训练数据（带噪声的）去测试拟合，不然就完全一致了（因为带噪声的才是真实情况）
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    # 可视化
    plot_kernel_reg(y_hat)
    # 可视化注意力汇聚情况
    show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='sorted training inputs',
                  ylabel='sorted testing inputs')
