import torch
from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    # AlexNet模型
    net = nn.Sequential(
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        # 输入层到隐藏层1，卷积操作
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        # 激活函数，引入非线性
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        # 隐藏层1到隐藏层2，卷积操作
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        # 池化操作，降维减参
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 展平操作，将多维数据展平为一维数据s
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        # 隐藏层到输出层，全连接层
        nn.Linear(6400, 4096),
        nn.ReLU(),
        # Dropout层，防止过拟合
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10)
    )
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: ', X.shape)

    # load data
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # start training
    print("start training:")
    lr, num_epochs = 0.01, 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device=d2l.try_gpu())
    d2l.plt.show()  # 可视化
