from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hidden, device = len(vocab), 256, d2l.try_gpu()

    num_inputs = vocab_size
    num_epochs, lr = 500, 1
    lstm_layer = nn.LSTM(num_inputs, num_hidden)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()
