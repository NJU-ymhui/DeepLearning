import torch
import math
from torch import nn
from d2l import torch as d2l


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hidden, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P, 准备之后填充正弦值和余弦值
        self.P = torch.zeros((1, max_len, num_hidden))
        # 套公式
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hidden, 2, dtype=torch.float32) / num_hidden)
        # 填充
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """将输入张量X与位置编码矩阵相加，使模型能够感知序列中元素的位置信息"""
        # 相加时只取X也有的维度大小部分，因为self.P可能与X的维度不同，所以只取X维度相同的部分，即从0取到X.shape[1]
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        # .to(X.device)确保所有参与运算的张量都在同一个设备上，避免跨设备运算导致的错误或性能问题。
        return self.dropout(X)


if __name__ == '__main__':
    # 定义序列长度和编码维度
    num_steps, num_encoding = 60, 32
    # 初始化位置编码对象
    pos_encode = PositionalEncoding(num_encoding, 0)
    pos_encode.eval()
    # 处理一个全零张量
    X = pos_encode(torch.zeros((1, num_steps, num_encoding)))
    # 获取处理后的张量
    P = pos_encode.P[:, :X.shape[1], :]
    # 可视化
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    d2l.plt.show()
