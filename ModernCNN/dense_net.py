import torch
from torch import nn
from d2l import torch as d2l


# 使用ResNet改良版的 批量规范化、激活和卷积 架构
# 实现该架构
def conv_block(input_channels, num_channels):
    """批量规范化、激活和卷积架构"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


# 实现稠密快
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


# 由于每个稠密块都会带来通道数的增加，使用过多会复杂化模型，而过渡层可以用来控制模型复杂度
# 通过1 * 1卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低复杂度
def transition_block(input_channels, num_channels):
    """
    :param input_channels: 输入通道数
    :param num_channels: 通道数
    :return: 过渡层
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


if __name__ == "__main__":
    # 创建稠密块
    blk = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)  # 4, 23, 8, 8

    # 对上述稠密块的输出使用通道数为10的过渡层, 高和宽减半
    blk = transition_block(23, 10)
    print(blk(Y).shape)  # 4, 10, 4, 4

    # DenseNet模型
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # num_channels为当前通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_block = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_block):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使其通道数量减半
        if i != len(num_convs_in_dense_block) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels //= 2
    # 创建DenseNet模型，和ResNet类似，最后接上全局汇聚层和全连接层来输出结果
    net = nn.Sequential(
        b1,
        *blks,
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),  # 展平
        nn.Linear(num_channels, 10)
    )

    # 训练模型
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()  # 可视化
