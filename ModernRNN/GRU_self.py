import torch
from d2l import torch as d2l


# 初始化模型参数
def get_params(vocab_size, num_hidden, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, num_hidden)),
            normal((num_hidden, num_hidden)),
            torch.zeros(num_hidden, device=device)
        )

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    # 注：附加梯度是一种集成学习技术，通过将多个弱学习器组合到一起，逐步提高模型的预测性能
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 定义模型

def init_gru_state(batch_size, num_hidden, device):
    return torch.zeros((batch_size, num_hidden), device=device),


def gru(inputs, state, params):
    # 解包参数，包括更新门、重置门和候选隐藏状态的相关参数
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    # 初始隐藏状态
    H, = state
    # 用于存储每个时间步的输出
    outputs = []
    # 遍历输入序列中的每个时间步
    for X in inputs:
        # 计算更新门的值
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        # 计算重置门的值
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        # 计算候选隐藏状态的值
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        # 更新隐藏状态
        H = Z * H + (1 - Z) * H_tilda
        # 计算输出
        Y = torch.matmul(H, W_hq) + b_q
        # 将当前时间步的输出添加到输出列表中
        outputs.append(Y)
    # 返回所有时间步的输出连接而成的张量和最终的隐藏状态
    return torch.cat(outputs, dim=0), (H, )


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 训练与预测
    vocab_size, num_hidden, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1  # 这些参数和之前一样
    model = d2l.RNNModelScratch(vocab_size, num_hidden, device, get_params, init_gru_state, gru)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    # 可视化困惑度
    d2l.plt.show()

