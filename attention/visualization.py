import torch
from d2l import torch as d2l


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    # 使用SVG格式显示图像，以获得更清晰的视觉效果
    d2l.use_svg_display()

    # 提取矩阵的行数和列数，用于后续的图形网格布局
    num_rows, num_cols, _, _ = matrices.shape

    # 创建一个图形和子图网格，根据矩阵的行数和列数进行布局
    # figsize参数用于设置图形的大小，sharex和sharey参数确保子图之间共享x和y轴的刻度，squeeze=False以保持子图数组的维度
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    # 遍历子图网格和矩阵，将矩阵可视化在相应的子图上
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # 在子图上绘制矩阵的图像
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            # 为最后一行的子图设置x轴标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 为第一列的子图设置y轴标签
            if j == 0:
                ax.set_ylabel(ylabel)
            # 如果提供了标题，则为子图设置标题
            if titles:
                ax.set_title(titles[j])

    # 在图形的右侧添加一个颜色条，用于表示矩阵值的含义
    # shrink参数用于调整颜色条的大小
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.show()


if __name__ == '__main__':
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
