from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hidden, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    num_epochs, lr, device = 500, 2, d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hidden, num_layers)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr * 1.0, num_epochs, device)
    d2l.plt.show()
