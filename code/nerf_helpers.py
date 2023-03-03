import math

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

# 根据相机内参及其位姿获取穿过图像的光线束的原点和方向
# 图像中像素image[i][j]对应的原点和方向是ray_origins[i][j]和ray_directions[i][j]
# 因此ray_origins的形状是(W,H,3)，ray_directions的形状是(W,H,3)
def get_ray_bundle(
        height, # 图像的像素高度
        width, # 图像的像素宽度
        focal, # 相机的焦距
        tform_cam2world # 相机的位姿，同时也是相机坐标系向世界坐标系变换的变换矩阵
):
    # 获取相机内参
    intrinsics = [focal, focal, 0.5, 0.5]

    # meshgrid + stack = 坐标
    # direction = [(i-0.5W)/f, -(j-0.5H)/f, -1]
    ii, jj = meshgrid_xy(
        torch.arange(width, dtype=tform_cam2world.dtype, device=tform_cam2world.device),
        torch.arange(height, dtype=tform_cam2world.dtype, device=tform_cam2world.device)
    )
    directions = torch.stack([
            (ii - width * intrinsics[2]) / intrinsics[0],
            -(jj - height * intrinsics[3]) / intrinsics[1],
            -torch.ones_like(ii),
        ], dim=-1,
    )

    # 将得到的各像素的o和d变换到世界坐标系
    # sum(A[..., None, :] * B, dim=-1)就是B与A的矩阵乘法
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)

    return ray_origins, ray_directions


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
):
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # 投影到NDC空间，推导见NeRF论文附录
    focal = [focal, focal]

    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1.0 / (W / (2.0 * focal[0])) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal[1])) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal[0]))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal[1]))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# 将巨大的张量拆分为成批的小张量
def get_minibatches(inputs, chunksize):
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def mse2psnr(mse):
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)