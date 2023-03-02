


import torch

# 获取编码函数，对positional_encoding函数进一步封装，输入仅为变量x
def get_embedding_function(
        num_encoding_functions, # 编码输入的数量
        include_input, # 编码中是否含原输入
        log_sampling # 是否使用对数采样进行编码
):
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )

# 位置编码函数
def positional_encoding(
        tensor, # 输入向量
        num_encoding_functions, # 编码输入的数量
        include_input, # 编码中是否含原输入
        log_sampling # 是否使用对数采样进行编码
):
    # 编码频率
    frequency_bands = 2.0 ** torch.linspace(
        0.0, num_encoding_functions - 1, num_encoding_functions,
        dtype=tensor.dtype, device=tensor.device
    ) if log_sampling else torch.linspace(
        2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions,
        dtype=tensor.dtype, device=tensor.device
    )

    # 编码结果
    encoding = [tensor] if include_input else []
    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    return torch.cat(encoding, dim=-1)
