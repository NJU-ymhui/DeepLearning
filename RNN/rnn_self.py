import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def get_params(vocab_size, num_hidden, device):
    """
    初始化神经网络的模型参数
    :param vocab_size 语言模型的输入输出来自同一个词表，因此他们具有相同的维度即词表大小
    :param num_hidden 隐藏层单元数，可调的超参数
    :param device
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hidden))
    W_hh = normal((num_hidden, num_hidden))
    b_h = torch.zeros(num_hidden, device=device)
    # 输出层参数
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hidden, device):
    """初始化时返回隐状态，返回值全0填充"""
    return torch.zeros((batch_size, num_hidden), device=device),


# rnn函数定义了如何在一个时间步内计算隐状态和输出
def rnn(inputs, state, params):
    # inputs形状: (时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状: (批量大小，词表大小)
    for X in inputs:
        # 使用tanh激活函数
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )


# 封装上述函数
class RNNModelScratch:
    """从零实现循环神经网络"""
    def __init__(self, vocab_size, num_hidden, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hidden = vocab_size, num_hidden
        self.params = get_params(vocab_size, num_hidden, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hidden, device)


def predict_ch8(prefix, num_pred, net, vocab, device):
    """预测字符串prefix后面的内容"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 预测num_pred步
    for _ in range(num_pred):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 把向量转化为索引
    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 索引转化为token


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 训练
def train_epoch_ch8(net, train_iter, loss, updator, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 如果state还没有初始化或者使用随机采样
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updator, torch.optim.Optimizer):
            updator.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updator.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用过mean方法
            updator(batch_size=1)
        metric.add(y.numel() * l, y.numel())
    # 第一个返回值是困惑度perplexity，用于衡量语言模型的性能
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 循环神经网络模型的训练函数
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updator = torch.optim.SGD(net.parameters(), lr)
    else:
        updator = lambda batch_size: d2l.sgd(net.params, lr, batch_size)  # 接受一个参数批量大小
    # 预测函数
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)  # 接受一个参数初始序列
    # 训练和预测
    perplexity, speed = -1, -1
    for epoch in range(num_epochs):
        perplexity, speed = train_epoch_ch8(net, train_iter, loss, updator, device, use_random)
        # print(f'perplexity:  {perplexity}')
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (perplexity,))
    d2l.plt.show()  # 可视化困惑度动态迭代结果
    print(f'perplexity:  {perplexity}, {speed} tokens / per second, on {device}')
    print("predict 'time traveller':")
    print(predict("time traveller"))
    print(predict("traveller"))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 之前一直将词元表示为一个索引, 但这样会使得模型难以学习(一个标量), 因此引入独热编码将词元映射为向量(互不相同的索引映射为互不相同的单位向量)
    print(F.one_hot(torch.tensor([0, 2, 3]), len(vocab)))
    # 每次采样的小批量数据形状是二维张量：（批量大小，时间步数）
    # one_hot函数将这样一个小批量数据转换成三维张量，张量的最后一个维度等于词表大小
    # 转换输入的维度，以便获得形状为（时间步数，批量大小，词表大小）的输出
    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, len(vocab)).shape)

    # 验证一下我们手搓的循环神经网络是否输出正确的形状
    num_hidden = 512
    net = RNNModelScratch(len(vocab), num_hidden, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    print()
    print(Y.shape)
    print(len(new_state))
    print(new_state[0].shape)

    # 不训练直接预测
    print("predict without training:\ntime traveller ...? ->")
    pred = predict_ch8("time traveller ", 10, net, vocab, d2l.try_gpu())  # 生成离谱的 预测结果
    print(pred)

    # 训练后再预测
    num_epochs, lr = 500, 1
    print("not random sample:")
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), False)  # 不使用随机采样
    print("random sample:")
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(), True)  # 使用随机采样
