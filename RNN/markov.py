import torch
from torch import nn
from d2l import torch as d2l


def generate_data(num):
    x = torch.arange(1, num + 1, dtype=torch.float32)
    y = torch.sin(0.01 * x) + torch.normal(0, 0.25, (num, ))
    return x, y


def init_weights(m):
    """初始化网络权重"""
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)


def get_net():
    """一个简单的多层感知机"""
    net = nn.Sequential(  # 一个有两个全连接层的多层感知机，使用ReLU激活函数
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, epochs, lr):
    """训练模型，与前面格式一致，不再赘述"""
    trainer = torch.optim.Adam(net.parameters(), lr)  # Adam优化器
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


if __name__ == "__main__" :
    t = 1000
    time, x = generate_data(t)
    d2l.plot(time, [x], 'time', 'x', legend=['x'], xlim=[1, t], figsize=(5, 2))
    d2l.plt.show()
    # 接下来，我们将这个序列转换为模型的特征－标签（feature‐label）对; features-labels就是前面讲过的的自变量-因变量，还记得吗？
    # 基于嵌入维度τ，我们将数据映射为数据对yt = xt 和xt = [xt−τ , . . . , xt−1]。
    # 这比我们提供的数据样本少了τ个，因为我们没有足够的历史记录来
    # 描述前τ个数据样本。一个简单的解决办法是：如果拥有足够长的序列就丢弃这几项；另一个方法是用零填充序列
    # 使用前600个“特征－标签”对进行训练
    tau = 4  # 取tau = 4
    # 初始化特征矩阵，其中t是时间序列的长度，tau是时间步的大小
    features = torch.zeros((t - tau, tau))
    # 遍历时间步，构建特征矩阵
    for i in range(tau):
        features[:, i] = x[i:t - tau + i]  # 列是时间步，行是数据序列
    labels = x[tau:].reshape((-1, 1))
    batch_size, n_train = 16, 600  # 用前600个数据对来训练
    # 开始训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
    # 下面使用一个相当简单的架构训练模型：
    # 一个拥有两个全连接层的多层感知机，ReLU激活函数，平方损失函数
    loss = nn.MSELoss(reduction='none')
    net = get_net()
    epochs, lr = 5, 0.01
    train(net, train_iter, loss, epochs, lr)
    # 接下来开始预测
    onestep_pred = net(features)  # net(...)应用模型进行预测, 这里net(features)对其中的每个值依次作用产生输出，因此可以视作单步预测
    d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_pred.detach().numpy()],
             xlabel='time', ylabel='x', xlim=[1, t], legend=['data', '1-step_pred'], figsize=(5, 2))
    d2l.plt.show()

    # 以上均为单步预测，下面使用我们的预测(而不是原始数据)进行多步预测
    multistep_pred = torch.zeros(t)
    # 用我们的预测数据填充
    multistep_pred[:n_train + tau] = x[:n_train + tau]
    # 利用之前的预测值进行多步预测
    # f(xt) = f(xt-1, xt-2, ..., xt-tau)
    for i in range(n_train + tau, t):
        multistep_pred[i] = net(multistep_pred[i - tau:i].reshape((1, -1)))  # 这步预测结果出来后会被后面继续使用
    d2l.plot([time, time[n_train + tau:]], [x.detach().numpy(), multistep_pred[n_train + tau:].detach().numpy()],
             xlabel='time', ylabel='x', xlim=[1, t], legend=['data', 'multi-step_pred'], figsize=(5, 2))
    d2l.plt.show()
    # 可以看到超过某个值后预测的效果很差，几乎趋于一个常数，这是由于错误的累积

    # 基于k = 1, 4, 16, 64，通过对整个序列预测的计算，让我们更仔细地看一下k步预测的困难
    max_steps = 64
    features = torch.zeros((t - tau - max_steps + 1, tau + max_steps))
    # features的列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau):
        features[:, i] = x[i:i + t - tau - max_steps + 1]
    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

    # 可视化1, 4, 16, 64步预测的结果
    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1: t - max_steps + i] for i in steps],
             [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
             legend=[f'{i}-step pred' for i in steps], xlim=[5, t],
             figsize=(5, 2))
    d2l.plt.show()
