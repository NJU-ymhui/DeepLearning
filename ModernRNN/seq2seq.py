import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
from encoder_decoder import Encoder
from encoder_decoder import Decoder
from encoder_decoder import EncoderDecoder
from machine_translation import load_data_nmt
from RNN.rnn_self import grad_clipping


class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hidden, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hidden, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X的形状：(batch_size, num_steps, embed_size_
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 若未提及状态，默认0
        output, state = self.rnn(X)
        # output的形状:(num_steps, batch_size, num_hidden)
        # state的形状:(num_layers, batch_size, num_hidden)
        return output, state


class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hidden, num_layer, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 定义词汇嵌入层，将词汇ID转换为词嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 定义GRU网络，用于处理序列数据
        # 输入维度为词嵌入向量维度embed_size与隐藏层单元数num_hidden之和
        # 隐藏层单元数为num_hidden，用于捕捉序列中的长期依赖关系
        # 设置多层GRU，num_layer表示GRU的层数
        # 添加dropout，用于在训练过程中防止过拟合
        self.rnn = nn.GRU(embed_size + num_hidden, num_hidden, num_layer, dropout=dropout)
        # 定义全连接层，将GRU的输出转换为词汇大小的预测值
        # 输入维度为GRU的隐藏层单元数num_hidden，输出维度为词汇表大小vocab_size
        self.dense = nn.Linear(num_hidden, vocab_size)

    def init_state(self, enc_outputs, *args):
        """
        :param enc_outputs: 编码器的输出
        :param args: 其余参数
        :return:
        """
        return enc_outputs[1]

    def forward(self, X, state):
        # X的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size, num_steps, vocab_size)
        # state的形状:(num_layers, batch_size, num_hidden)
        return output, state


# 下面将通过计算交叉熵损失函数来进行优化
# 首先定义一个遮蔽函数通过零值化来屏蔽不相关预测
def sequence_mask(X, valid_len, value=0):
    """屏蔽序列中不相关的项"""
    # 获取当前批次中序列的最大长度
    max_len = X.size(1)
    # 创建一个形状为(batch_size, max_len)的掩码，其中valid_len对应位置为True，其余为False
    mask = torch.arange((max_len), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    # 将不符合掩码条件的元素替换为指定的value
    X[~mask] = value
    # 返回应用掩码后的序列数据
    return X


# 定义损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred形状：(batch_size, num_steps, vocab_size)
    # label形状：(batch_size, num_steps)
    # valid_length形状：(batch_size,)

    def forward(self, pred, label, valid_len):
        # 初始化与标签形状相同的权重张量，初始权重都为1
        weights = torch.ones_like(label)
        # 应用sequence_mask以根据有效长度对权重进行掩码，超出有效长度的部分权重设为0
        weights = sequence_mask(weights, valid_len)
        # 设置reduction参数为'none'，确保损失函数为每个元素返回一个未减少的损失值
        self.reduction = 'none'
        # 调用父类的forward方法计算未加权的损失，调整预测数据的维度以适应损失函数的要求
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 将未加权的损失与掩码权重相乘，然后在序列维度上计算加权损失的平均值
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# 训练序列到序列学习模型
def train_seq2seq(net, train_iter, lr, num_epochs, target_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loas = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in train_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([target_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            # 这里不做l.sum()会报RuntimeError: Boolean value of Tensor with more than one value is ambiguous
            grad_clipping(net, l.sum())
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]}, {metric[1] / timer.stop()} tokens / sec on {device}')
    d2l.plt.show()  # 可视化损失曲线


# 预测
# 为了采用一个接着一个词元的方式预测输出序列，每个解码器当前时间步的输入都来自前一时间步的预测词元
def predict_seq2seq(net, src_sentence, src_vocab, target_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    net.eval()  # 评估模式
    # 将源句子转换为词元序列，并添加序列结束符
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    # 记录有效长度，用于处理padding
    end_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 确保序列长度不超过num_steps，不足则填充
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    # 通过编码器编码源句子
    enc_outputs = net.encoder(enc_X, end_valid_len)
    # 初始化解码器状态
    dec_state = net.decoder.init_state(enc_outputs, end_valid_len)
    # 添加批量轴
    # 解码器的输入开始于开始符号
    dec_X = torch.unsqueeze(
        torch.tensor([target_vocab['<bos>']], dtype=torch.long, device=device), dim=0
    )
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # 使用解码器生成下一个词元
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        # 挤压批量轴，获取预测的词元ID
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 若序列结束词元被预测，输出序列的生成就完成了
        if pred == target_vocab['<eos>']:
            break
        # 累积预测的词元序列
        output_seq.append(pred)
    # 将词元ID序列转换为目标句子
    return ' '.join(target_vocab.to_tokens(output_seq)), attention_weight_seq


# 预测序列的评估
def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    # 将序列分割成词 token
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    # 计算预测序列和标签序列的长度
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算精度的初步分数部分
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 遍历不同的n-gram长度
    for n in range(1, k + 1):
        # 初始化匹配数量和标签子序列的字典
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 构建标签子序列的n-gram并计数
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        # 在预测序列中查找匹配的n-gram
        for i in range(len_pred - n + 1):
            # 如果在标签序列中找到匹配，则增加匹配数量
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] += 1
        # 根据匹配数量和n-gram长度更新分数
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    # 返回最终的BLEU分数
    return score


if __name__ == "__main__":
    # 实例化一个Seq2SeqEncoder对象，用于编码序列
    # 参数说明：
    # vocab_size: 词汇表大小，表示输入数据的唯一词汇数量
    # embed_size: 嵌入层大小，表示将词汇嵌入到多少维度的向量空间
    # num_hidden: 隐藏层单元数量，决定了模型的复杂度
    # num_layers: RNN的层数，多层可以提高模型的表达能力
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hidden=16, num_layers=2)
    encoder.eval()  # 评估模式
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print("encoder:")
    print('output shape:', output.shape, 'state shape:', state.shape)

    # 使用与上面编码器相同的超参数来实例化解码器
    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hidden=16, num_layer=2)
    decoder.eval()
    state = decoder.init_state(encoder(X))
    output, state = decoder(X, state)  # 获得输出，并更新state
    print("decoder:")
    print('output shape:', output.shape, 'state shape:', state.shape)

    # 定义损失函数
    loss = MaskedSoftmaxCELoss()
    print("loss demo:")
    print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))

    # 现在在机器翻译数据集上，我们可以创建和训练一个循环神经网络“编码器-解码器”模型用于序列到序列的学习
    embed_size , num_hidden, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
    train_iter, src_vocab, target_vocab = load_data_nmt(batch_size, num_steps)  # 用d2l的库会报编码错误
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hidden, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(target_vocab), embed_size, num_hidden, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, target_vocab, device)

    # 利用训练好的“编码器-解码器”模型，将几个英语句子翻译为法语，并计算BLEU最终结果
    print("start translating:")
    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, target_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
