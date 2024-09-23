import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 为一个完整的循环神经网络模型定义一个RNNModule类
# 由于rnn_layer只包含隐藏的循环层，因此还需要创建一个单独的输出层
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hidden = self.rnn.hidden_size
        # 如果RNN是双向的，num_directions应该是 2，否则是 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hidden, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hidden * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(num_steps * batch_size, num_hidden)
        # 它的输出形状是(num_steps * batch_size, vocab_size)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        # LSTM: 长短期记忆网络：一种特殊的循环神经网络
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hidden), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hidden),
                                device=device), torch.zeros((self.num_directions * self.rnn.num_layers,
                                                             batch_size, self.num_hidden), device=device))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hidden = 256
    # 定义模型
    rnn_layer = nn.RNN(len(vocab), num_hidden)
    # 初始化隐状态
    state = torch.zeros((1, batch_size, num_hidden))
    print(state.shape)

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)  # rnn_layer就是之前的net

    # 训练与预测
    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device=device)
    predict = d2l.predict_ch8("time traveller", 10, net, vocab, device)
    print("predict of 'time traveller' without training:")
    print(predict)  # 这样得到的是一个胡扯的结果，因为没有训练
    print("start training:")
    num_epochs, lr = 500, 0.1
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device=device)
    d2l.plt.show()  # 可视化困惑度
