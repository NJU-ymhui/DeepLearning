import torch
import collections
import re
from torch import nn
from d2l import torch as d2l


def read_time_machine():
    with open(d2l.download("time_machine"), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  # 只要英文字母且全小写


def tokenize(lines, token="word"):
    """
    将输入行拆分为词元
    :param lines: 输入的文本行列表
    :param token: 次元类型
    :return: 拆分后的列表
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError("Invalid token flag: " + token)


# 建立一个词表，记录词元到数字的映射
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按词元出现频率排序，降序
        counter = count_corpus(tokens)
        self._tokens_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元索引为0
        self.idx2token = ['<ink>'] + reserved_tokens  # 索引到词元
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}  # 词元到索引
        for token, freq in self._tokens_freq:
            if freq < min_freq:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self):
        """返回词汇表中词汇的数量"""
        return len(self.idx2token)

    def __getitem__(self, tokens):
        """将一个或多个词汇转换为对应的索引，若词汇不存在，则返回未知词汇标识"""
        # 若tokens不是列表或元组，则直接返回该词汇的索引或未知词汇标识
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, self.unk)
        # 若tokens是列表或元组，则逐个转换为索引
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """将一个或多个索引转换为对应的词汇"""
        # 若indices不是列表或元组，则直接返回该索引对应的词汇
        if not isinstance(indices, (list, tuple)):
            return self.idx2token[indices]
        # 若indices是列表或元组，则逐个转换为词汇
        return [self.idx2token[index] for index in indices]

    @property  # unk可以像属性一样被访问，而不需要调用方法
    def unk(self):
        return 0

    @property
    def token_freq(self):
        return self._tokens_freq


def count_corpus(tokens):
    """统计词元频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    """整合所有功能，返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, token='char')  # 次元类型改为char
    vocab = Vocab(tokens)
    # 因为数据集中的每一个文本行不一定是一个句子或者段落
    # 所以展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        # 如果限定了最大tokens的数量，我们就只取前max行
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    # 从时光机器文本中读取数据
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
    lines = read_time_machine()
    print(len(lines))
    print(lines[0])
    print(lines[114])

    # 词元化
    tokens = tokenize(lines)
    for token in tokens[:10]:
        print(token)

    # 用上面拿到的数据集构建词表，看几个高频词及其索引
    vocab = Vocab(tokens)
    print("high frequency token and its index:")
    print(list(vocab.token2idx.items())[:10])

    # 现在就可以把每一行文本转化成索引序列了
    print("line text -> indices:")
    for i in [0, 10]:
        print(tokens[i], '->', vocab[tokens[i]])

    # 验证整合的功能
    print("check all functions in one:")
    tokens, vocab = load_corpus_time_machine()  # 接受词元索引列表和词表
    print((len(tokens), len(vocab)))
