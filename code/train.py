import os
import time
from datetime import datetime

import torch
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from yacs.config import CfgNode
from PIL import Image

from load_dataset import load_data
from nerf_helpers import get_ray_bundle, meshgrid_xy, mse2psnr, img2mse, cast_to_image, \
    create_parser, create_nerf
from train_utils import run_one_iter_of_nerf


def main():
    # 设置调试时返回更多信息
    torch.autograd.set_detect_anomaly(True)

    # 命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 读取配置文件
    cfg = None
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # 加载其它基本配置
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    print("Starting data loading.")
    images, poses, expressions, bboxs, i_split, hwf, render_poses = load_data(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.testskip
    )
    # 获取划分的训练集、验证集、测试集索引
    i_train, i_val, i_test = i_split
    # 获取相机内参
    H, W, focal = int(hwf[0]), int(hwf[1]), hwf[2]
    # 将图像从RGBA转为RGB，公式为targetRGB = sourceRGB * sourceA + BGcolor * (1 - sourceA)
    if cfg.nerf.train.white_background:  # BGcolor = 1
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    print("Done data loading.")

    # 设置NeRF模型
    model_coarse, model_fine, encode_position_fn, encode_direction_fn = create_nerf(cfg)
    model_coarse.to(device)
    if hasattr(cfg.models, "fine"):
        model_fine.to(device)

    # # ??? ablation
    # blur_background = False
    #
    # regularize_latent_codes = True  # True to add latent code LOSS, false for most experiments

    # 设置背景
    trainable_background = False
    fixed_background = True
    if trainable_background: # 可训练背景
        print("Creating trainable background.")
        # 背景初始化为训练集图像的平均值
        with torch.no_grad():
            avg_img = torch.mean(images[i_train], axis=0)
            # ??? 模糊背景图
            # ------------
            background = torch.tensor(avg_img, device=device)
        background.requires_grad = True
    elif fixed_background: # 固定的真实背景
        print("Loading GT background.")
        background = Image.open(os.path.join(cfg.dataset.basedir, 'bg', '00001.png'))
        background.thumbnail((H, W))
        background = torch.from_numpy(np.array(background).astype(np.float32)).to(device)
        background = background / 255
    else:
        background = None

    # 设置潜在代码
    use_latent_codes = True
    if use_latent_codes:
        latent_codes = torch.zeros((len(i_train), 32), device=device)
        latent_codes.requires_grad = True
    else:
        latent_codes = None

    # 生成射线重要性采样图
    ray_importance_sampling_maps = []
    p = 0.9
    for i in i_train:
        # 提取当前图像的边界框
        bbox = bboxs[i]
        # 边界框内的像素概率设置为p，边界框外的像素概率设置为1-p
        probs = np.zeros((H, W))
        probs.fill(1 - p)
        probs[bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
        # 概率归一化
        probs = (1 / probs.sum()) * probs
        # 压扁后加入射线重要性采样图中
        ray_importance_sampling_maps.append(probs.reshape(-1))

    # 设置优化器
    trainable_parameters = list(model_coarse.parameters())
    if model_fine:
        trainable_parameters += list(model_fine.parameters())
    if trainable_background:
        trainable_parameters.append(background)
    if use_latent_codes:
        trainable_parameters.append(latent_codes)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(trainable_parameters, lr=cfg.optimizer.lr)

    # 生成训练日志
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id, datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())

    # 加载已有检查点
    start_iter = 0
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if checkpoint["background"]:
            background = torch.nn.Parameter(checkpoint['background'].to(device))
        if checkpoint["latent_codes"]:
            latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(device))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # 开始训练
    print("Starting loop.")
    for i in trange(start_iter, cfg.experiment.train_iters):
        # 开启训练模式
        model_coarse.train()
        if model_fine:
            model_fine.train()

        # 从训练集中选出一张图像及其位姿、表情、潜在代码、重要性采样图信息
        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        expression_target = expressions[img_idx].to(device)
        latent_code = latent_codes[img_idx].to(device) if use_latent_codes else None
        ray_importance_sampling_map = ray_importance_sampling_maps[img_idx]

        # 计算穿过图像所有像素[i][j]的光线束的原点(W,H,3)和方向(W,H,3)（世界坐标系下）
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)

        # meshgrid + stack = 坐标
        coords = torch.stack(meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)), dim=-1)
        # 压扁为(HW, 2)
        coords = coords.reshape((-1, 2))
        # 根据重要性采样图随机选出图像中的像素点坐标，共选择cfg.nerf.train.num_random_rays个像素坐标
        select_inds = np.random.choice(coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, p=ray_importance_sampling_map)
        select_inds = coords[select_inds]
        # 获取选出的像素坐标对应的原点o和方向d
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]  # (cfg.nerf.train.num_random_rays, 3)
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]  # (cfg.nerf.train.num_random_rays, 3)
        # 获取GT图像及(GT背景/训练背景)中选出的像素坐标的颜色 (512,512,3) ==> (cfg.nerf.train.num_random_rays, 3)
        target_ray_values = img_target[select_inds[:, 0], select_inds[:, 1], :]
        background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (trainable_background or fixed_background) else None

        # 计算渲染后的像素颜色
        rgb_coarse, rgb_fine, weights = run_one_iter_of_nerf(
            H,  # 图像高度
            W,  # 图像宽度
            focal,  # 相机焦距
            model_coarse,  # 粗模型
            model_fine,  # 细模型
            ray_origins,  # 射线原点
            ray_directions,  # 射线方向
            cfg,  # 从配置文件中读取的配置
            mode="train", # train或validation
            encode_position_fn=encode_position_fn,  # 位置编码函数
            encode_direction_fn=encode_direction_fn,  # 方向编码函数
            expressions=expression_target,  # GT表情系数
            background_prior=background_ray_values,  # GT背景像素颜色
            latent_code=latent_code  # 潜在代码
        )

        # 计算损失函数
        coarse_loss = torch.nn.functional.mse_loss(rgb_coarse[..., :3], target_ray_values[..., :3])
        fine_loss = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_ray_values[..., :3]) if rgb_fine is not None else 0.0
        latent_code_loss = torch.norm(latent_code) * 0.0005
        background_loss = torch.mean(
            torch.nn.functional.mse_loss(
                background_ray_values[..., :3], target_ray_values[..., :3], reduction='none'
            ).sum(1) * weights
        ) * 0.001

        loss = coarse_loss
        if rgb_fine is not None:
            loss = loss + fine_loss
        psnr = mse2psnr(loss.item())
        if trainable_background:
            loss = loss + background_loss
        if use_latent_codes:
            loss = loss + latent_code_loss * 10

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 学习率衰减
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        # 输出训练指标
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(f"[TRAIN] Iter: {str(i)} Loss: {str(loss.item())} BG Loss: {str(background_loss.item())} PSNR: {str(psnr)} LatentReg: {str(latent_code_loss.item())}")

        # 生成训练日志
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)
        if trainable_background:
            writer.add_scalar("train/bg_loss", background_loss.item(), i)
        if use_latent_codes:
            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)

        # 验证阶段
        if i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write("[VAL] =======> Iter: " + str(i))

            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None

                loss = 0
                for img_idx in i_val:
                    # 获取验证集图像及其位姿、表情、潜在代码信息
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    expression_target = expressions[img_idx].to(device)
                    latent_code = torch.zeros(32).to(device)

                    # 计算穿过验证集图像所有像素的光线束的原点和方向（世界坐标系下）
                    ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
                    ray_origins = ray_origins.view((-1, 3))
                    ray_directions = ray_directions.view((-1, 3))

                    # 计算渲染后的完整图像
                    rgb_coarse, rgb_fine, weights = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        expressions=expression_target,
                        background_prior=background.view(-1, 3) if (trainable_background or fixed_background) else None,
                        latent_code=torch.zeros(32).to(device) if use_latent_codes else None,
                        validation_image_shape=img_target.shape
                    )

                    # 计算损失函数
                    target_ray_values = img_target
                    coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                    loss = loss + coarse_loss
                    if rgb_fine is not None:
                        fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                        loss = loss + fine_loss

                # 计算验证指标，生成验证日志
                loss /= len(i_val)
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_image("validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i)
                if rgb_fine is not None:
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                    writer.add_image("validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i)
                writer.add_image("validation/img_target", cast_to_image(target_ray_values[..., :3]),i)
                if trainable_background or fixed_background:
                    writer.add_image("validation/background", cast_to_image(background[..., :3]), i)
                    writer.add_image("validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW')
                tqdm.write(f"Validation loss: {str(loss.item())} Validation PSNR: {str(psnr)} Time: {str(time.time() - start)}")

        # 保存检查点
        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": model_fine.state_dict() if model_fine else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss - background_loss,
                "psnr": psnr,
                "background": background.data if (trainable_background or fixed_background) else None,
                "latent_codes": latent_codes.data if use_latent_codes else None
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


if __name__ == "__main__":
    main()
