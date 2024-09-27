import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        # kernel_size (tuple): 卷积核的大小，用于初始化权重矩阵。
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))  # 随机初始化权重
        self.bias = nn.Parameter(torch.zeros(1))  # 为偏置赋初值

    def forward(self, x):
        # 应用卷积操作并加上偏置项
        return corr2d(x, self.weight) + self.bias


if __name__ == "__main__":
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))

    # 边缘检测
    X = torch.ones((6, 8))  # 构造一个6 * 8的黑白像素图象，0黑1白
    X[:, 2:6] = 0
    print(X)
    # 接下来构造一个长为1宽为2的卷积核，当相邻元素相同时，输出0
    K = torch.tensor([[1.0, -1.0]])  # [1, 0] * K = 1, [0, 1] * K = -1
    # 执行互相关运算
    Y = corr2d(X, K)
    # 输出中1为白色到黑色的边缘, -1为黑色到白色的边缘
    print(Y)  # 根据输出发现这种方法只能检测出垂直边缘，水平边缘消失了
    # 更直观的感受一下
    print(corr2d(X.T, K))

    # 构造一个二维卷积层，它有1个输出通道和形状为(1, 2)的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # 卷积核权重存在这里
    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高、宽）
    # 批量大小和通道都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    # 准备训练
    num_epochs, lr = 10, 0.03
    print("before training:")
    print("weight:", conv2d.weight.data.reshape((1, 2)))
    for i in range(num_epochs):
        Y_hat = conv2d(X)
        l = (Y - Y_hat) ** 2
        conv2d.zero_grad()
        l.sum().backward()  # 先梯度归零，再反向传播
        # 更新权重
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        print(f'epoch {i + 1}, loss {l.sum():.5f}')  # 可视化损失变化

    # 看一下迭代后的权重如何
    print("after training:")
    print("weight:", conv2d.weight.data.reshape((1, 2)))
