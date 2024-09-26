import os
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt


def read_data_nmt():
    """载入\"英语-法语\"数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """对原始数据做一些预处理，如将不间断空格替换为一个空格，小写替换大写，单词和标点之间插入空格"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写替换大写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else
           char for i, char in enumerate(text)]
    return ''.join(out)


# 词元化
def tokenize_nmt(text, num_examples=None):
    """词元化\"英语-法语\"数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# 绘制每个文本序列所包含的词元数量的直方图
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    # 使用原生pyplot 绘制直方图
    plt.figure(figsize=(5, 3))
    _, _, patches = plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)
    plt.show()


# 加载数据集
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for  l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']
    ) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    reserved_tokens = ['<pad>', '<bos>', '<eos>']
    # 从文件中读取原始数据，并进行预处理
    text = preprocess_nmt(read_data_nmt())
    # 对预处理后的数据进行分词，并按需限制样本数量
    source, target = tokenize_nmt(text, num_examples)
    # 构建源语言和目标语言的词表，最小频率设为2
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=reserved_tokens)
    target_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=reserved_tokens)
    # 将分词后的文本序列转换为索引数组和有效长度数组
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    target_array, target_valid_len = build_array_nmt(target, target_vocab, num_steps)
    # 组合所有数据数组，以便加载到迭代器中
    data_arrays = [src_array, src_valid_len, target_array, target_valid_len]
    # 创建并返回数据迭代器，以及源语言和目标语言的词表
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, target_vocab


if __name__ == "__main__":
    d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                               '94646ad1522d915e7b0f9296181140edcf86a4f5')
    # 加载数据
    raw_text = read_data_nmt()
    print("raw data:")
    print(raw_text[:75])

    # 数据预处理
    print("after preprocessing:")
    text = preprocess_nmt(raw_text)
    print(text[:75])

    # 词元化
    source, target = tokenize_nmt(text)
    print("after tokenizing:")
    print(source[:6])
    print(target[:6])

    show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)

    # 词表
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bcs>', '<eos>'])
    print("vocab size:")
    print(len(src_vocab))

    # 截断或填充文本
    print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

    # 加载迭代器和词表
    train_iter, src_vocab, target_vocab = load_data_nmt(batch_size=2, num_steps=8)
    # 可视化一部分数据
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break
