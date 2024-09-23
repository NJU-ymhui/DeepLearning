import torch
import random


def seq_fata_iter_random(corpus, batch_size, num_steps):
    """
    :param corpus:
    :param batch_size: 每个小批量中子序列样本的数量
    :param num_steps: 每个子序列中预定义的时间步数
    :return:
    """
    # 从语料库中随机选择一个片段作为开始, 切片内容包括num_steps - 1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 计算基于当前语料库长度和序列长度能够生成的序列数量
    # 减去1，是因为我们需要考虑标签
    num_sequences = (len(corpus) - 1) // num_steps
    # 创建一个列表，包含所有序列的起始索引，即长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_sequences * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    # 定义一个辅助函数，根据给定的起始位置从语料库中提取序列
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_sequences // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        # initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]  # 从打乱顺序的起始索引列表中获取当前批次的起始索引
        # 根据当前批次中每个序列的起始索引，创建X（输入序列）和Y（目标序列）
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # 生成并提供输入和目标序列的张量表示
        yield torch.tensor(X), torch.tensor(Y)


if __name__ == '__main__':
    # 生成一个0 ~ 34的序列，并设置批量大小 = 2，时间步数 = 5
    # 这样可以生成(35 - 1) // 5 = 6个特征-标签子序列对
    seq = list(range(35))
    i = 1
    print("每个小批量中有两个子序列对:")
    for X, Y in seq_fata_iter_random(seq, batch_size=2, num_steps=5):
        print(f"第%d个\"特征-标签\"子序列对小批量" % i)
        print(f'X: {X}, \nY: {Y}')
        i += 1
