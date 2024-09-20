import math

import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """
    在最后一个轴上掩蔽元素来执行softmax操作
    :param X: 三维张量
    :param valid_lens: 一维或二维张量 
    :return: softmax操作后的结果
    """  # 最后一个维度代表特征数，因此在最后一个维度上操作即对每个特征都操作，且为了处理序列中不同元素的有效长度不同的情况
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:  # 1D张量，扩展为2D张量，重复元素即为复制这个向量
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])  # shape[1]为三维张量X第二个轴的元素个数
        else:
            valid_lens = valid_lens.reshape(-1)
        # 对序列进行掩码，将超过有效长度(valid_lens)的部分设置为极小值(value=-1e6), 以便后续处理中这些位置的影响会被忽略
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # reshape保证形状不变，dim=-1在最后一个维度上计算softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, queries_size, keys_size, num_hidden, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.w_q = nn.Linear(queries_size, num_hidden, bias=False)  # 用于key的输入层到隐藏层变换
        self.w_k = nn.Linear(keys_size, num_hidden, bias=False)  # 用于query的输入层到隐藏层变换
        self.w_v = nn.Linear(num_hidden, 1, bias=False)  # 隐藏层到输出层，即打分
        self.dropout = nn.Dropout(dropout)  # 传入暂退概率
        
    def forward(self, queries, keys, values, valid_lens):
        # 在多层感知机中对queries和keys作从输入层到隐藏层的变换
        queries = self.w_q(queries)
        keys = self.w_k(keys)
        # 假设queries: (batch_sz, n, d)->(batch_sz, n, 1, d); keys: (batch_sz, m, d)->(batch_sz, 1, m, d)
        # 广播机制求和后: (batch_sz, n, m, d)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)  # queries在第二维加入新维度，keys在第一维加入新维度，可以利用广播机制相加
        features = torch.tanh(features)  # 激活
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # 初始输出的一个scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]  # d取queries最后一个维度的大小, 即查询和键的长度
        # 计算评分
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # 套公式, 不过要注意keys的转置不能用.T，会报错(?)，要用transpose
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    # 掩蔽softmax操作

    # # 随机生成一个有两个矩阵的张量
    # print(masked_softmax(torch.rand(2, 3, 4), torch.tensor([2, 3])))  # 第一个矩阵有效长度2，第二个矩阵有效长度3
    # # 也可以为矩阵的每一行也指定长度
    # print(masked_softmax(torch.rand((2, 3, 4)), torch.tensor([[1, 2, 1], [3, 4, 2]])))

    # 加性注意力

    # # 准备数据
    # # 初始化查询、键、值和有效长度
    # # 查询: 使用正态分布生成的随机数组
    # # 键: 一组全为1的数组
    # # 值: 从0到39的连续数字的数组，每个数字重复两次
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    #
    # attention = AdditiveAttention(keys_size=2, queries_size=20, num_hidden=8, dropout=0.1)
    # # 将模型设置为评估模式
    # attention.eval()
    # # 计算注意力权重
    # # 注意：这里直接调用attention(queries, keys, values, valid_lens)，而不是使用.forward()方法
    # # 因为在PyTorch中，如果模型没有处于训练模式，通常可以直接调用模型实例来进行预测或推理
    # print(attention(queries, keys, values, valid_lens))  # 不要错用成attention.forward()
    # # 可视化注意力
    # d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='keys', ylabel='queries')
    # d2l.plt.show()

    # 缩放点积注意力

    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()  # 不进入训练模式，而是评估模式
    print(attention(queries, keys, values, valid_lens))
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='keys', ylabel='queries')
    d2l.plt.show()
