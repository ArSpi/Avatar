import torch

from nerf_helpers import ndc_rays, get_minibatches, cumprod_exclusive, sample_pdf


def volume_render_radiance_field(
    radiance_field, # 一批射线上所有采样点的颜色和密度 (num_rays, num_coarse, 4)
    depth_values, # 一批射线上所有采样点的深度z_vals (num_rays, num_coarse)
    ray_directions, # 穿过各像素点的射线的方向向量 (num_rays, 3)
    radiance_field_noise_std=0.0, # 体积渲染时需要添加到辐射场中的噪音的标准偏差
    white_background=False, # 是否使用白色作为背景的底色
    background_prior = None # 各像素点的背景颜色
):
    # 计算同一条射线上的各采样点的深度差，最后添加1e10代表积分到无穷远处 (num_rays, num_coarse)
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],  # (num_rays, num_coarse-1)
            one_e_10.expand(depth_values[..., :1].shape),  # (num_rays, 1)
        ),
        dim=-1,
    )

    # 计算同一条射线上的各采样点的深度差的实际欧式距离 (num_rays, num_coarse)
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    # 提取神经辐射场中的一批射线上各采样点的rgb值，其中最后的背景采样点不通过sigmoid
    if background_prior is not None:
        rgb = torch.sigmoid(radiance_field[:, :-1, :3]) # rgb(num_rays, num_coarse-1, 3)
        rgb = torch.cat((rgb, radiance_field[:, -1, :3].unsqueeze(1)), dim=1) # rgb(num_rays, num_coarse, 3)
    else:
        rgb = torch.sigmoid(radiance_field[..., :3])

    # 对神经辐射场中的一批射线上各采样点的sigmoid值添加均值为0、方差为radiance_field_noise_std的高斯白噪声
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )  # 生成满足均值为0、方差为1的高斯白噪声
            * radiance_field_noise_std
        )
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)

    # 计算一系列同质媒介的alpha值 alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-sigma_a * dists)

    # 计算一系列同质媒介的rgb权重值 weight_i = T_i * alpha_i
    # where T_i = exp(-\sum_{j=1}^{i-1}{sigma_i * delta_i}) = \prod_{j=1}^{i-1}{1 - alpha_i}
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    # 对各段同质媒介进行权重求和，得到图像中一批像素的rgb值（与一批射线对应）
    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)  # dim=-2是num_coarse采样点

    # 添加背景的白色底色
    acc_map = weights.sum(dim=-1) # 计算不透明度
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, weights


def run_network(
        network_fn,  # 模型网络
        pts,  # 一批射线上所有采样点的位置坐标
        ray_batch,  # 一批射线的ro,rd,near,far
        chunksize,  # chunksize: 2048
        embed_fn,  # 位置编码函数
        embeddirs_fn,  # 方向编码函数
        expressions,  # GT表情系数
        latent_code  # 潜在代码
):
    # 将一批射线上所有采样点的位置坐标压扁 (num_rays, num_coarse, 3) ==> (num_rays*num_coarse, 3)
    pts_flat = pts.reshape((-1, pts.shape[-1]))

    # 对位置坐标进行位置编码 (num_rays*num_coarse, 3) ==> (num_rays*num_coarse, 3*(1+6*2))
    embedded = embed_fn(pts_flat)
    # 对方向进行方向编码
    viewdirs = ray_batch[..., None, -3:]
    input_dirs = viewdirs.expand(pts.shape)
    input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    # 合并两种编码
    embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    # 将嵌入向量划分批次
    batches = get_minibatches(embedded, chunksize=chunksize)

    # 将嵌入向量与表情、潜在向量一起分批加入模型网络中，得到预测的采样点的颜色rgb和体积密度sigma (num_rays*num_coarse, 4)
    preds = [network_fn(batch, expressions, latent_code) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)

    # 恢复radiance_field的形状 (num_rays*num_coarse, 4) ==> (num_rays, num_coarse, 4)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )

    del embedded, input_dirs_flat

    return radiance_field


def predict_and_render_radiance(
        ray_batch,  # 成批的射线束，包含ro,rd,near,far (num_rays, 8)
        model_coarse, # 粗分辨率网络
        model_fine, # 细分辨率网络
        cfg, # 配置文件
        mode, # train或validation
        encode_position_fn, # 位置编码函数
        encode_direction_fn, # 方向编码函数
        expressions, # GT表情系数
        background_prior, # GT背景/训练背景
        latent_code # 潜在代码
):
    # 获取批量射线的条数
    num_rays = ray_batch.shape[0]  # chunksize: 2048
    # 从ray_batch中拆分出ro,rd,near,far
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # 对于粗分辨率网络，从0到1进行均匀采样
    t_vals = torch.linspace(0.0, 1.0, getattr(cfg.nerf, mode).num_coarse, dtype=ro.dtype, device=ro.device)
    # 将从0到1的均匀采样线性映射到从near到far的均匀采样
    z_vals = (near * (1.0 - t_vals) + far * t_vals) if not getattr(cfg.nerf, mode).lindisp else (1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals))
    # 将均匀采样的采样点扩展到所有射线上
    z_vals = z_vals.expand([num_rays, getattr(cfg.nerf, mode).num_coarse])
    # 对采样点进行扰动，所有射线上的采样点扰动相同
    if getattr(cfg.nerf, mode).perturb:
        # 对每个采样点规定其上界upper和下界lower
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # 在上界和下界之间随机扰动
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand

    # 获取一批射线上的所有采样点的位置坐标 pts = o + td (num_rays, num_coarse, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    # 对射线采样点处的颜色和密度进行预测 radiance_field(num_rays, num_coarse, 4)
    radiance_field = run_network(
        model_coarse,  # 粗分辨率网络
        pts,  # 一批射线上的所有采样点的位置坐标 (num_rays, num_coarse, 3) == (射线数, 射线上的采样数, 采样点的位置坐标)
        ray_batch,  # 小批量的rays (chunksize, 8)
        getattr(cfg.nerf, mode).chunksize,  # chunksize: 2048
        encode_position_fn,  # 位置编码函数
        encode_direction_fn,  # 方向编码函数
        expressions,  # GT表情系数
        latent_code  # 潜在代码
    )

    # 将最后一个采样点的颜色更改成背景的颜色
    if background_prior is not None:
        radiance_field[:, -1, :3] = background_prior

    # 将神经辐射场渲染出二维图像
    rgb_coarse, weights = volume_render_radiance_field(
        radiance_field,  # 射线采样点的颜色和密度 (num_rays, num_coarse, 4)
        z_vals,  # 采样点深度（从near到far）
        rd,  # 穿过各像素点的射线的方向向量 (num_rays, 3)
        radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,  # 体积渲染时需要添加到辐射场中的噪音的标准偏差
        white_background=getattr(cfg.nerf, mode).white_background,  # 是否使用白色作为背景的底色
        background_prior=background_prior  # 各像素点的背景颜色
    )

    # 根据粗分辨率网络的渲染结果训练细分辨率网络
    rgb_fine = None
    if getattr(cfg.nerf, mode).num_fine:
        # 获取每段同质媒介的中点
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 根据权重获取采样点 (num_rays, num_fine)
        z_samples = sample_pdf(
            z_vals_mid,  # 同质媒介的中点
            weights[..., 1:-1],  # 同质媒介的rgb权重
            getattr(cfg.nerf, mode).num_fine,  # 采样点数目
            det=(getattr(cfg.nerf, mode).perturb == 0.0)  # 是否对采样点扰动
        )
        z_samples = z_samples.detach()

        # 合并粗采样点和细采样点，将每条射线上的采样点升序重排 (num_rays, num_coarse + num_fine)
        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # 获取一批射线上各采样点的位置坐标 (num_rays, num_coarse + num_fine, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        # 对射线采样点处的颜色和密度进行预测 radiance_field(num_rays, num_coarse + num_fine, 4)
        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(cfg.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            expressions,
            latent_code
        )

        # 将最后一个采样点的颜色更改成背景的颜色
        if background_prior is not None:
            radiance_field[:, -1, :3] = background_prior

        # 将神经辐射场渲染出二维图像
        rgb_fine, weights = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background,
            background_prior=background_prior
        )

    # 注意返回的权重值是细分辨率网络训练出的权重值
    return rgb_coarse, rgb_fine, weights[:, -1]


def run_one_iter_of_nerf(
        height,  # 图像的像素高度
        width,  # 图像的像素宽度
        focal_length,  # 相机的焦距
        model_coarse,  # 粗分辨率模型
        model_fine,  # 细分辨率模型
        ray_origins,  # 图像中选出的像素对应的射线原点 (cfg.nerf.train.num_random_rays, 3)
        ray_directions,  # 图像中选出的像素对应的射线方向 (cfg.nerf.train.num_random_rays, 3)
        cfg,  # 配置文件
        mode,  # train或validation
        encode_position_fn,  # 位置编码函数
        encode_direction_fn,  # 方向编码函数
        expressions,  # GT表情系数
        background_prior,  # GT背景/训练背景 (cfg.nerf.train.num_random_rays, 3)
        latent_code,  # 潜在代码
        validation_image_shape=None  # 验证过程应返回图像的形状（训练过程只需要返回像素集）
):
    # 产生rays，0-2维是ro，3-5维是rd，6维是near，7维是far (cfg.nerf.train.num_random_rays, 8)
    ro, rd = ray_origins, ray_directions if cfg.dataset.no_ndc else ndc_rays(
        height, width, focal_length, 1.0, ray_origins, ray_directions
    ) # NDC空间用于无界场景，对于blender等有界场景不使用NDC空间
    near = cfg.dataset.near * torch.ones_like(rd[..., :1])
    far = cfg.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)  # (cfg.nerf.train.num_random_rays, 8)

    # 将ray_directions归一化，得到单位方向向量viewdirs，并加入rays中
    viewdirs = ray_directions
    viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)  # (cfg.nerf.train.num_random_rays, 3)
    rays = torch.cat((rays, viewdirs), dim=-1) # (cfg.nerf.train.num_random_rays, 11)

    # 将rays拆分成小批量的射线束
    chunksize = getattr(cfg.nerf, mode).chunksize
    batches = get_minibatches(
        rays,  # 各射线的(ro,rd,near,far)
        chunksize=chunksize  # chunksize: 2048
    )
    # 将background_prior拆分成小批量的像素集（一个像素正对应一条射线）
    background_prior = get_minibatches(
        background_prior,  # GT背景像素颜色
        chunksize=chunksize  # chunksize: 2048
    ) if background_prior is not None else background_prior  # GT背景像素颜色

    # 获取预测的图像各像素的颜色和权重参数
    pred = [
        predict_and_render_radiance(
            batch,  # 小批量的rays (chunksize, 11)
            model_coarse,  # 粗网络
            model_fine,  # 细网络
            cfg,  # 从配置文件中读取的配置
            mode,  # "train"
            encode_position_fn=encode_position_fn,  # 位置编码函数
            encode_direction_fn=encode_direction_fn,  # 方向编码函数
            expressions=expressions,  # GT表情系数
            background_prior=background_prior[i] if background_prior is not None else background_prior,
            # 小批量的rays对应的小批量的background_prior (chunksize, 3)
            latent_code=latent_code,  # 潜在代码
        )
        for i, batch in enumerate(batches)
    ]

    # 拆分整理批量的返回值
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]

    # 如果是验证过程，不应返回像素集，而是返回整张图像
    if mode == "validation":
        num_pixels = validation_image_shape[0] * validation_image_shape[1]
        shapes = [validation_image_shape, validation_image_shape, (num_pixels,)]
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, shapes)
        ]
        return tuple(synthesized_images)

    # return rgb_coarse, rgb_fine, weights[:, -1] 这里的weight是细分辨率网络训练出的权重值
    return tuple(synthesized_images)