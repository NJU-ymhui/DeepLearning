import math
import torch
import numpy as np
from d2l import torch as d2l
from timer import Timer
from matplotlib import pyplot as plt


def compare_forloop_tensor():
    # 对比for循环和张量加
    a = torch.arange(100000)
    b = torch.arange(100000)
    res = torch.zeros(100000)
    clock = Timer()
    for i in range(len(a)):
        res[i] += a[i] + b[i]
    print(f'use for loop: {clock.stop(): .5f} sec')
    clock.start()
    c = a + b
    print(f'use tensor: {clock.stop(): .5f} sec')


if __name__ == '__main__':
    compare_forloop_tensor()
