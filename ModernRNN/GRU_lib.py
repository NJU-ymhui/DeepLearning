from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, device = len(vocab), d2l.try_gpu()
    num_epochs, lr = 500, 1

    num_inputs, num_hidden = vocab_size, 256
    gru_layer = nn.GRU(num_inputs, num_hidden)
    model = d2l.RNNModel(gru_layer, vocab_size)  # 实例化GRU
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()

