import torch
from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    # 初始化隐藏层神经元数量、头的数量
    num_hidden, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hidden, num_hidden, num_hidden, num_hidden, num_heads, dropout=0.5)
    print(attention.eval())  # 不训练

    # 初始化批量大小，查询的数量，有效词元长度
    batch_size, num_queries, valid_lens = 6, 6, torch.tensor([1, 1, 4, 5, 1, 4])
    X = torch.ones((batch_size, num_queries, num_hidden))  # 6个矩阵，每个矩阵都是6 * 100的
    tmp = attention(X, X, X, valid_lens)
    print(tmp)
    print(tmp.shape)
