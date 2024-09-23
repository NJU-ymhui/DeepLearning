import torch
import random


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """顺序分区策略生成小批量子序列"""
    # 从随机偏移量开始划分序列
    # 生成一个随机偏移量，用于乱序数据
    offset = random.randint(0, num_steps)
    # 计算基于批次大小可处理的令牌数量
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # 从乱序后的数据中创建输入序列Xs
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    # 从乱序后的数据中创建目标序列Ys，相比Xs向后移动了一个位置
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # 将序列Xs和Ys重塑为批次大小，以便于训练
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 计算批次的数量
    num_batches = Xs.shape[1] // num_steps
    # 遍历所有序列，生成批次数据
    for i in range(0, num_steps * num_batches, num_steps):
        # X是当前批次的输入序列，长度为num_steps
        X = Xs[:, i: i + num_steps]
        # Y是当前批次的目标序列，长度为num_steps
        Y = Ys[:, i: i + num_steps]
        # 产出当前批次的输入序列X和目标序列Y
        yield X, Y


if __name__ == "__main__":
    # 数据配置与随机采样策略一致
    seq = list(range(35))
    for X, Y in seq_data_iter_sequential(seq, batch_size=2, num_steps=5):
        print(f'X: {X},\nY: {Y}')
