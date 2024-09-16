import torch
from d2l import torch as d2l


def relu():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)  # ReLU激活函数
    # .detach()方法用于创建一个新的Tensor，该Tensor从当前计算图中分离出来，但仍指向相同的数据
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))  # 绘制图像, x和y的数据通过.detach()方法从计算图中分离，避免梯度计算
    d2l.plt.show()
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
    d2l.plt.show()


def sigmoid():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
    d2l.plt.show()
    # 求导数
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
    d2l.plt.show()


def tanh():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.tanh(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    d2l.plt.show()
    # 求导
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
    d2l.plt.show()


if __name__ == '__main__':
    # relu()
    # sigmoid()
    tanh()