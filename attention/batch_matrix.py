import torch


if __name__ == '__main__':
    a = torch.ones((2, 1, 4))
    b = torch.ones((2, 4, 6))
    print(torch.bmm(a, b).shape)

    # 注意力背景下的批量矩阵乘法
    weights = torch.ones((2, 10)) * 0.1
    print("weights:")
    print(weights)
    print("after unsqueeze:")
    print(weights.unsqueeze(1))  # 通过添加维度从而可以利用批量矩阵乘法
    values = torch.arange(20, dtype=torch.float32).reshape((2, 10))
    print("values:")
    print(values)
    print("after unsqueeze:")
    print(values.unsqueeze(-1))  # 通过添加维度从而可以利用批量矩阵乘法
    print("calculate:")
    print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))
    print(torch.mm(weights, values.T))
