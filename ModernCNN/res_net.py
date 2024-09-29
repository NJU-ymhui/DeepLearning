import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 实现残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            # 如果使用1 * 1卷积层，添加通过1 * 1卷积调整通道和分辨率
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            # 不使用1 * 1卷积层，在应用ReLU函数之前，将输入添加到输出
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# ResNet使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。
# 第一个模块的通道数同输入通道数一致, 由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽
# 之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
# 下面实现这个模块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


if __name__ == "__main__":
    # 当输入和输出形状一致时
    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)

    blk = Residual(3, 6, use_1x1conv=True, strides=2)
    print(blk(X).shape)

    # ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、步幅为2的7 × 7卷积层后，接步幅为2的3 × 3的最大汇聚层。
    # 不同之处在于ResNet每个卷积层后增加了批量规范化层
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 10)
    )

    # 观察不同模块的输入输入形状是如何变化的
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "output shape: ", X.shape)

    # 训练模型
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()  # 可视化
