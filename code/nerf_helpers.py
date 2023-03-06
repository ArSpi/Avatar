import math

import torch


# 获取编码函数，对positional_encoding函数进一步封装，输入仅为变量x
def get_embedding_function(
        num_encoding_functions,  # 编码输入的数量
        include_input,  # 编码中是否含原输入
        log_sampling  # 是否使用对数采样进行编码
):
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )


# 位置编码函数
def positional_encoding(
        tensor,  # 输入向量
        num_encoding_functions,  # 编码输入的数量
        include_input,  # 编码中是否含原输入
        log_sampling  # 是否使用对数采样进行编码
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
        height,  # 图像的像素高度
        width,  # 图像的像素宽度
        intrinsics,  # 相机的内参
        tform_cam2world  # 相机的位姿，同时也是相机坐标系向世界坐标系变换的变换矩阵
):
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
    return [inputs[i: i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def mse2psnr(mse):
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


# [a, b, c, d] ==> [1, a, a*b, a*b*c]
def cumprod_exclusive(tensor):
    dim = -1  # 仅处理最后一维

    # 累乘
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.0

    return cumprod


def sample_pdf(
        bins,  # 同质媒介的中点深度 (num_rays, num_coarse-1)
        weights,  # 同质媒介的rgb权重 (num_rays, num_coarse-1)
        num_samples,  # 采样点数目
        det  # 是否对采样点扰动
):
    # # 将权重归一化，得到概率密度函数 probability density function(PDF) (num_rays, num_coarse-1)
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # # 构建累积分布函数 cumulative distribution function(CDF) (num_rays, num_coarse-1)
    cdf = torch.cumsum(pdf, dim=-1)  # 累加
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # 前补0

    # # 计算0到1之间的采样点 (num_rays, num_samples)
    if det:  # 在0到1之间均匀采样
        u = torch.linspace(0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:  # 在0到1之间随机采样
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], dtype=weights.dtype, device=weights.device)

    u = u.contiguous()
    cdf = cdf.contiguous()

    # # 在cdf曲线的纵轴上进行细采样 (num_rays, num_samples)
    # 具体来说，对于u[i]，返回ind[i]，满足cdf[ind[i]-1] <= u[i] < cdf[ind[i]]
    # 就是对u[i]“上取整”到cdf[ind[i]]，如果正好有u[i]=cdf[idx]，则ind[i]=idx+1
    # 权重更大处，cdf曲线更陡，采样点更容易集中
    inds = torch.searchsorted(cdf.detach(), u, right=True)

    # # 获取细采样点对应的cdf曲线上的两个线性插值点
    # 获取细采样点u在cdf上的“下取整”（用来作为cdf的索引），如果正好有u[i]=cdf[idx]，则ind[i]=idx below = max{0, inds - 1}
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    # 获取细采样点u在cdf上的“上取整”（用来作为cdf的索引），如果正好有u[i]=cdf[idx]，则ind[i]=idx above = min{len(cdf)-1, inds}
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # 合并两者 inds_g = [below, above] (num_rays, num_samples, 2)
    inds_g = torch.stack((below, above), dim=-1)

    # # 获取两个线性插值点对应的cdf值和深度值
    # 准备cdf曲线和深度值bins的匹配形状(num_rays, num_samples, num_coarse-1)
    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    # 获取细采样点的below和above索引对应的cdf值 cdf_g[i][j][k] = cdf[i][j][inds_g[i][j][k]] (num_rays, num_samples, 2)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    # 获取细采样点的below和above索引对应的深度值 bins_g[i][j][k] = bins[i][j][inds_g[i][j][k]] (num_rays, num_samples, 2)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # # 使用两个线性插值点的cdf值获取线性插值系数
    # 获取细采样点u对应的同质媒介的pdf值
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # 如果denom小于1e-5，将其更改为1（针对uniform采样时采到1的情况）
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # 获取线性插值系数，实际上有t = (u - cdf[below]) / (cdf[above] - cdf[below])
    t = (u - cdf_g[..., 0]) / denom

    # 对两个线性插值点的深度值进行线性插值，得到最终的深度采样点 (num_rays, num_samples)
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
