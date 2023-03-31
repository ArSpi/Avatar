import math

import torch

from train_utils import Mode
from nerf_helpers import custom_meshgrid, meshgrid_xy

import raymarching


class NeRFRenderer(torch.nn.Module):
    def __init__(self,
                 cfg,
                 dataset,
                 # bound=1,
                 # density_scale=1,
                 # min_near=0.2,
                 # density_thresh=0.01
                 ):
        self.cfg = cfg
        self.dataset = dataset
        self.bound = self.cfg.bound
        self.cascade = 1 + math.ceil(math.log2(self.bound))
        self.grid_size = 128
        self.density_scale = self.cfg.density_scale
        self.min_near = self.cfg.min_near
        self.density_thresh = self.cfg.density_thresh

        aabb_train = torch.FloatTensor([-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        density_grid = torch.zeros([self.cascade, self.grid_size ** 3])  # [CAS, H*H*H]
        density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8)  # [CAS*H*H*H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.num_update_density = 0

        # 计数采样点的数量和射线的数量
        samples_rays_counter = torch.zeros(16, 2, dtype=torch.int32)  # 16 is hardcoded for averaging...
        self.register_buffer('samples_rays_counter', samples_rays_counter)
        self.mean_sample_count = 0
        self.active_sample_step = 0

    def get_rays(self, index, num_sample_rays, ray_importance_sampling_map):
        # 生成横坐标和纵坐标
        ii, jj = meshgrid_xy(
            torch.arange(self.dataset.W, device=self.dataset.device),
            torch.arange(self.dataset.H, device=self.dataset.device)
        )
        ii = ii.reshape([self.dataset.W * self.dataset.H]) + 0.5  # [WH]
        jj = jj.reshape([self.dataset.W * self.dataset.H]) + 0.5  # [WH]
        # 确定射线数
        num_sample_rays = min(num_sample_rays, self.dataset.W * self.dataset.H)  # N
        # 在图像上随机选择像素
        select_inds = torch.multinomial(ray_importance_sampling_map, num_sample_rays, replacement=False)
        # 获取像素的横坐标和纵坐标
        ii, jj = torch.gather(ii, dim=-1, index=select_inds), torch.gather(jj, dim=-1, index=select_inds)

        # 确定相机内参
        fx, fy, cx, cy = self.dataset.intrinsics
        # 计算射线方向
        zs = torch.ones_like(ii)
        xs = (ii - cx) / fx * zs
        ys = (jj - cy) / fy * zs
        directions = torch.stack((xs, ys, zs), dim=-1)
        # 射线方向归一化
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        # 获取射线原点和射线方向（世界坐标系下）
        rays_d = directions @ self.dataset.poses[index][:3, :3].transpose(-1, -2)
        rays_o = self.dataset.poses[index][:3, 3].expand(rays_d.shape)

        return rays_o, rays_d, select_inds

    @torch.no_grad()
    def mark_untrained_grid(self, dataset, S=64):
        # poses: [B, 4, 4]
        # intrinsics: [4]

        poses = dataset.poses
        intrinsics = dataset.intrinsics

        B = dataset.num_data

        fx, fy, cx, cy = intrinsics

        # 创建tensor([0,...,grid_size-1])，然后按照S=64切割
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        mask_aabb = torch.zeros_like(self.density_grid)  # 用来标记超出aabb框范围的网格
        mask_cam = torch.zeros_like(self.density_grid)  # 用来标记超出摄像机范围的网格

        poses = poses.to(dataset.device)

        for xs in X:  # [0,...,S-1],...,[kS,...,last]
            for ys in Y:  # [0,...,S-1],...,[kS,...,last]
                for zs in Z:  # [0,...,S-1],...,[kS,...,last]

                    # construct points
                    # 就是torch.meshgrid，只不过按照版本分为不同的函数调用
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    # 创建三维坐标，代表当前坐标范围块内的三维坐标点
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    # 计算三维莫顿码，将N个三维坐标点[0,S)转化为具有特定系数的N个一维坐标点[0,S^3)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    # 将coords由[0,grid_size-1]标准化到[-1,1]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        # 选取切片2 ** cas
                        bound = min(2 ** cas, self.bound)
                        # 获得尺度比例值
                        half_grid_size = bound / self.grid_size
                        # 将体素网格坐标转换到bound的尺度，相当于乘以half_grid_size*(self.grid_size - 1)
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # 标记超出aabb框范围的网格
                        mask_min = (cas_world_xyzs >= (self.aabb_train[:3] - half_grid_size)).sum(-1) == 3
                        mask_max = (cas_world_xyzs <= (self.aabb_train[3:] + half_grid_size)).sum(-1) == 3
                        mask_aabb[cas, indices] += (mask_min & mask_max).reshape(-1)

                        # 标记超出摄像机范围的网格
                        # split batch to avoid OOM 内存溢出
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # 将世界坐标系的体素网格坐标转换到相机坐标系的体素网格坐标（需要w2c矩阵）
                            # pose是c2w矩阵，需要转置，然而坐标是横向量，也需要转置，因此两者相乘的最终形式不转置
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3]  # [S, N, 3] [图像数，图像上的射线数，三维坐标]

                            # 相机坐标系下的当前坐标会被训练到
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0  # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).bool().reshape(-1)  # [N]

                            # update count
                            mask_cam[cas, indices] += mask
                            head += S

        # mark untrained grid as -1
        self.density_grid[((mask_aabb == 0) | (mask_cam == 0))] = -1

        print(
            f'[mark untrained grid] {((mask_aabb == 0) | (mask_cam == 0)).sum()} from {self.grid_size ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        tmp_grid = - torch.ones_like(self.density_grid)

        # 全部更新体素网格密度
        if self.num_update_density < 16:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:

                        # 构建[-1,1]内标准化后的体素网格坐标
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                           dim=-1)  # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long()  # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            # 选取切片2 ** cas
                            bound = min(2 ** cas, self.bound)
                            # 获得尺度比例值
                            half_grid_size = bound / self.grid_size
                            # 将坐标转换到bound的尺度，相当于乘以half_grid_size*(self.grid_size - 1)
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # 为坐标添加[-hgs, hgs]内的扰动
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # 查询添加扰动后的坐标对应的密度值
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            # 放大体素网格顶点的密度，使密度更加尖锐，可以提升性能
                            sigmas *= self.density_scale
                            # 存储密度
                            tmp_grid[cas, indices] = sigmas

        # 部分更新体素网格密度
        else:
            N = self.grid_size ** 3 // 4  # H * H * H / 4
            for cas in range(self.cascade):
                # 随机选择N个体素网格顶点的三维坐标
                coords = torch.randint(0, self.grid_size, (N, 3),
                                       device=self.density_bitfield.device)  # [N, 3], in [0, 128)
                # 获取这些坐标的莫顿编码
                indices = raymarching.morton3D(coords).long()  # [N]
                # 获取非零密度的索引，density_grid[cas]是当前分辨率下各网格顶点的密度
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)  # [Nz]
                # 在非零密度的网格顶点中随机选择N个顶点
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long,
                                          device=self.density_bitfield.device)
                # 使用rand_mask随机选择N个非零密度的网格顶点
                occ_indices = occ_indices[rand_mask]  # [Nz] --> [N], allow for duplication
                # 从这些顶点的莫顿编码中获取顶点的三维坐标
                occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]

                # 将随机选择的N个顶点与随机选择的N个非零密度顶点合并
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # 将三维坐标标准化
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                # 选取切片2 ** cas
                bound = min(2 ** cas, self.bound)
                # 获得尺度比例值
                half_grid_size = bound / self.grid_size
                # 将坐标转换到bound的尺度，相当于乘以half_grid_size*(self.grid_size - 1)
                cas_xyzs = xyzs * (bound - half_grid_size)
                # 为坐标添加[-hgs, hgs]内的扰动
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # 查询添加扰动后的坐标对应的密度值
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                # 放大体素网格顶点的密度，使密度更加尖锐，可以提升性能
                sigmas *= self.density_scale
                # 存储密度
                tmp_grid[cas, indices] = sigmas

        # ema update
        # 选择原体积密度和更新后体积密度都大于等于0的部分
        # 原先是-1的不更新（已经在初始化时确定不训练）；更新后小于0的不合理，不更新
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        # 更新先前选取的部分，更新为更大的体积密度
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        # 计算所有体素网格密度的平均值（密度是-1的部分当作0计算）
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()  # -1 regions are viewed as 0 density.

        # 更新次数计数
        self.num_update_density += 1

        # 将密度阈值更新为更小值（原先的阈值和平均密度相比）
        density_thresh = min(self.mean_density, self.density_thresh)
        # 在bitfield中将密度大于阈值的网格顶点对应的标志位置1
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        # 更新mean_sample_count和active_sample_step
        # mean_sample_count是samples_rays_counter的第一维（采样点数量）的平均
        total_step = min(16, self.active_sample_step)
        if total_step > 0:
            self.mean_sample_count = int(self.samples_rays_counter[:total_step, 0].sum().item() / total_step)
        self.active_sample_step = 0

    def render(self, model, mode, rays_o, rays_d, expression, background_prior, latent_code):
        prefix = rays_o.shape[:-1]

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if mode is Mode.TRAIN else self.aabb_infer, self.min_near)

        if mode is Mode.TRAIN:
            sample_ray_counter = self.samples_rays_counter[self.active_sample_step % 16]
            sample_ray_counter.zero_()
            self.active_sample_step += 1

            # --------------------* 采样过程 *--------------------- #
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(
                rays_o,  # 射线原点
                rays_d,  # 射线方向
                self.bound,  # 场景边界
                self.density_bitfield,  # 体素网格的标志位
                self.cascade,  # 多分辨率级
                self.grid_size,  # 每一级分辨率体素网格的边长的网格数
                nears,  # 近边界
                fars,  # 远边界
                sample_ray_counter,  # 计数采样点的数量和射线的数量
                self.mean_sample_count,  #
                self.cfg.perturb,  #
                128,  #
                self.cfg.force_all_rays,  # False
                self.cfg.dt_gamma,  #
                self.cfg.max_steps  #
            )

            # --------------------* 运行模型 *--------------------- #
            sigmas, rgbs = model(xyzs, dirs, expression, latent_code)

            # --------------------* 渲染过程 *--------------------- #
            weights_sum, depth, rgb_pred = raymarching.composite_rays_train(
                sigmas,
                rgbs,
                deltas,
                rays,
                self.cfg.T_thresh
            )

            rgb_pred = rgb_pred + (1 - weights_sum).unsqueeze(-1) * background_prior
            rgb_pred = rgb_pred.view(*prefix, 3)
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            depth = depth.view(*prefix)

        # # 如果是验证过程，不应返回像素集，而是返回整张图像
        # if mode == "validation":
        #     shapes = [validation_image_shape, validation_image_shape, validation_image_shape[:-1]]
        #     synthesized_images = [
        #         image.view(shape) if image is not None else None
        #         for (image, shape) in zip(synthesized_images, shapes)
        #     ]
        #     return tuple(synthesized_images)

        # return rgb_coarse, rgb_fine, weights[:, -1] 这里的weight是细分辨率网络训练出的背景的权重值，用于前景与背景的分割
        # return tuple(synthesized_images)

        return rgb_pred