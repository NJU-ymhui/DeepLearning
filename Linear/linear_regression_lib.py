import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn  # 导入神经网络

"""
使用现有的库实现线性回归
"""


# 读取数据可以直接使用现有的data框架里的api
def load_data(data_tuple, batch_size, is_train=True):
    """实现一个pytorch迭代器"""
    # *用于解包，将data_tuple解包为多个参数，并作为独立的参数传入TensorDataset
    dataset = data.TensorDataset(*data_tuple)  # 将输入的元组数据转换为TensorDataset
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建DataLoader，设置批量大小为batch_size


if __name__ == '__main__':
    # actual params and offset
    actual_w = torch.tensor([1.14, 5.14])
    actual_b = 1.91981
    data_size = 1145

    batch_size = 10
    features, labels = d2l.synthetic_data(actual_w, actual_b, data_size)  # 生成数据集也使用现成库
    data_iter = load_data((features, labels), batch_size)
    print(next(iter(data_iter)))  # 看一下效果，从可迭代对象data_iter中获取一个迭代器，并调用next()方法从该迭代器中取出下一个元素

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))  # 选择现有库nn中的线性模型，输入维度为2，输出维度为1，即将两个输入特征映射成一个输出结果

    # init model
    net[0].weight.data.normal_(0, 0.01)  # 初始化权重参数, 均值为0，方差为0.01
    net[0].bias.data.fill_(0)  # 初始化偏置参数，0

    # 定义损失函数
    loss = nn.MSELoss()  # 均方误差损失函数

    # 定义优化算法
    opt = torch.optim.SGD(net.parameters(), lr=0.03)  # 随机梯度下降，学习率为0.03

    # train
    num_epochs = 3
    for epoch in range(num_epochs):
        l = 0.0
        for X, y in data_iter:
            l = loss(net(X), y)  # 计算损失
            opt.zero_grad()  # 梯度清零
            l.backward()  # 反向传播计算梯度
            opt.step()  # 更新参数
        print(f'epoch {epoch + 1}, loss {float(l):f}')
    print("real:")
    print(actual_w)
    print(actual_b)
    print("predict:")
    print(net[0].weight.data)
    print(net[0].bias.data)
