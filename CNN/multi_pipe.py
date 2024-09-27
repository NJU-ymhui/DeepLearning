import torch
from torch import nn
from d2l import torch as d2l


# 多输入
def corr2d_multi_in(X, K):
    # 先遍历X和K的第0个维度(通道维度)，再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


# 多输出
def corr2d_multi_in_out(X, K):
    # 遍历K的第0个维度，每次都把一个卷积层应用于X(执行互相关运算)，然后把结果收集起来
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 1 * 1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


if __name__ == "__main__":
    print("multi in:")
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))

    print("multi out:")
    K = torch.stack((K, K + 1, K + 2), 0)
    print(K.shape)
    print(corr2d_multi_in_out(X, K))

    print("1 * 1 correlation:")
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
    print(float(torch.abs(Y1 - Y2).sum()))
