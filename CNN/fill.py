import torch
from torch import nn


# 定义一个计算卷积层的函数
# 为函数初始化卷积层权重，并对输入和输出提高和缩减相应的维度
def comp_conv2d(conv2d, X):
    # (1, 1)表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度批量大小和通道数
    return Y.reshape(Y.shape[2:])


if __name__ == "__main__":
    # 这里每边都填充了1行或1列，因此共添加2行或2列
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    Y = comp_conv2d(conv2d, X)
    print(Y.shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    Y = comp_conv2d(conv2d, X)
    print(Y.shape)

    # 步幅 2
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    Y = comp_conv2d(conv2d, X)
    print(Y.shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    Y = comp_conv2d(conv2d, X)
    print(Y.shape)
