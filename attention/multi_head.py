import torch
import math
from torch import nn
from d2l import torch as d2l


# 为了能使多个头并行计算，定义两个转置函数方便MultiHeadAttention类调用
# 具体来说，transpose_output反转了transpose_qkv的操作
def transpose_qkv(X, num_heads):
    """为了多头注意力的并行计算而改变形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hidden / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hidden / num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hidden / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)  # 为什么写成X.permute(0, 2, 1, 3).reshape(X.shape[0], X.shape[1], -1)不行


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = d2l.DotProductAttention(dropout=dropout)
        self.w_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.w_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.w_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.w_o = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.num_heads = num_heads

    def forward(self, queries, keys, values, valid_lens):
        # 使用不同的线性变换（w_q, w_k, w_v）处理查询、键和值之后，再根据多头注意力机制的需要，重新排列这些处理后的数据
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，以此类推
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads, 查询的个数, num_hidden / num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状: (batch_size, 查询的个数, num_hidden)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


if __name__ == '__main__':
    # 测试MultiHeadAttention类
    num_hidden, num_heads = 100, 5
    attention = MultiHeadAttention(num_hidden, num_hidden, num_hidden, num_hidden, num_heads, dropout=0.5)
    print(attention.eval())
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hidden))
    Y = torch.ones((batch_size, num_kvpairs, num_hidden))
    print(attention(X, Y, Y, valid_lens).shape)
