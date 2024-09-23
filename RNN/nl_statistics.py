import torch
from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    lines = d2l.read_time_machine()
    tokens = d2l.tokenize(lines)
    corpus = [token for line in tokens for token in line]
    vocab = d2l.Vocab(corpus)
    print(vocab.token_freqs[:10])
    frequencies = [freq for token, freq in vocab.token_freqs]
    d2l.plot(frequencies, xlabel='token: x', ylabel='n(x)', xscale='log', yscale='log')
    d2l.plt.show()

    # 查看一下二元语法出现的概率
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = d2l.Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    # 再来看一下三元组
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    # 对比一下三种组合的出现概率图
    bigram_freq = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freq = [freq for token, freq in trigram_vocab.token_freqs]
    d2l.plot([frequencies, bigram_freq, trigram_freq], xlabel='token: x', ylabel='n(x)', xscale='log', yscale='log',
             legend=['single', 'double', 'triple'])
    d2l.plt.show()
