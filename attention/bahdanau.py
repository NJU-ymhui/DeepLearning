import torch
from torch import nn
from d2l import torch as d2l


class AttentionDecoder(d2l.Decoder):
    """解码器接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    # TODO
    def __init__(self):
        # TODO
        return


if __name__ == '__main__':
    # TODO
    a = 1
    