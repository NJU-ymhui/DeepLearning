import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from d2l import torch as d2l


def normal(x, mu, sigma):
    # 正态分布公式
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


def gauss_distribution():
    x = torch.arange(-7, 7, 0.01)  # 从-7 到 7，步长为0.01
    mu_sigma = [(0, 1), (0, 2), (3, 1)]  # 三种均值和标准差组合
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in mu_sigma], figsize=(4.5, 2.5), xlabel='x', ylabel='p(x)')
    plt.show()


if __name__ == '__main__':
    gauss_distribution()