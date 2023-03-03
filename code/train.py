import argparse
import os
import time
from datetime import datetime

import torch
import torchvision
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from yacs.config import CfgNode
from PIL import Image

from load_dataset import load_data
import models
from nerf_helpers import get_embedding_function, get_ray_bundle, meshgrid_xy, mse2psnr
from train_utils import run_one_iter_of_nerf


def create_nerf(cfg):
    # 初始化位置嵌入函数
    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,  # 位置编码的采样频率的数量
        include_input=cfg.models.coarse.include_input_xyz,  # 位置编码中是否包括输入的向量
        log_sampling=cfg.models.coarse.log_sampling_xyz,  # 位置编码的样本点是否以对数形式采样
    )

    # 初始化方向嵌入函数
    encode_direction_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,  # 方向编码的采样频率的数量
        include_input=cfg.models.coarse.include_input_dir,  # 方向编码中是否包括输入的向量
        log_sampling=cfg.models.coarse.log_sampling_dir,  # 方向编码的样本点是否以对数形式采样
    )

    # 初始化粗分辨率网络
    model_coarse = getattr(models, cfg.models.coarse.type)(
        # 模型结构
        num_layers=cfg.models.coarse.num_layers,  # 模型的层数
        hidden_size=cfg.models.coarse.hidden_size,  # 隐藏层的大小
        skip_connect_every=cfg.models.coarse.skip_connect_every,  # 每隔多少层跳跃连接
        # 向量嵌入
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,  # 位置编码的采样频率的数量
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,  # 方向编码的采样频率的数量
        include_input_xyz=cfg.models.coarse.include_input_xyz,  # 位置编码中是否包括输入的向量
        include_input_dir=cfg.models.coarse.include_input_dir,  # 方向编码中是否包括输入的向量
        # 潜在代码
        latent_code_dim=cfg.models.coarse.latent_code_dim
    )

    # 初始化细分辨率网络
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,  # 模型的层数
            hidden_size=cfg.models.fine.hidden_size,  # 隐藏层的大小
            skip_connect_every=cfg.models.fine.skip_connect_every,  # 每隔多少层跳跃连接

            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,  #
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,  #
            include_input_xyz=cfg.models.fine.include_input_xyz,  #
            include_input_dir=cfg.models.fine.include_input_dir,  #

            latent_code_dim=cfg.models.fine.latent_code_dim
        )

    return model_coarse, model_fine, encode_position_fn, encode_direction_fn


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, help="path to load saved checkpoint."
    )
    return parser


def main():
    # 设置调试时返回更多信息
    torch.autograd.set_detect_anomaly(True)

    # 命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 读取配置文件
    cfg = None
    with open(args.config, 'r') as f:
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
    # train_background = False
    # supervised_train_background = False
    # blur_background = False
    #
    # train_latent_codes = True
    # disable_expressions = False  # True to disable expressions
    # disable_latent_codes = False  # True to disable latent codes
    # fixed_background = True  # Do False to disable BG
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
        background = Image.open(os.path.join(cfg.dataset.basedir, 'bg', '00000.png'))
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
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())

    # 加载已有检查点
    start_iter = 0
    if os.path.exists(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint)
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
        # 获取GT图像及(GT背景/训练背景)中选出的像素坐标的颜色
        target_ray_values = img_target[select_inds[:, 0], select_inds[:, 1], :]  # (512,512,3) ==> (cfg.nerf.train.num_random_rays, 3)
        background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (trainable_background or fixed_background) else None

        # 计算渲染后的图像
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
        fine_loss = torch.nn.functional.mse_loss(rgb_fine[..., :3], target_ray_values[..., :3]) if rgb_fine else 0.0
        latent_code_loss = torch.norm(latent_code) * 0.0005 if use_latent_codes else 0.0
        background_loss = torch.mean(
            torch.nn.functional.mse_loss(
                background_ray_values[..., :3], target_ray_values[..., :3], reduction='none'
            ).sum(1) * weights
        ) * 0.001 if trainable_background else 0.0

        loss = coarse_loss + fine_loss
        psnr = mse2psnr(loss.item())
        loss += background_loss
        loss += latent_code_loss * 10

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
        if rgb_fine:
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
                # rgb_coarse, rgb_fine = None, None
                # target_ray_values = None
                #
                # loss = 0
                # for img_idx in i_val[:2]:
                #     img_target = images[img_idx].to(device)
                #     pose_target = poses[img_idx, :3, :4].to(device)
                #     ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
                #     rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
                #         H,
                #         W,
                #         focal,
                #         model_coarse,
                #         model_fine,
                #         ray_origins,
                #         ray_directions,
                #         cfg,
                #         mode="validation",
                #         encode_position_fn=encode_position_fn,
                #         encode_direction_fn=encode_direction_fn,
                #         expressions=expression_target,
                #         background_prior=background.view(-1, 3) if (trainable_background or fixed_background) else None,
                #         latent_code=torch.zeros(32).to(device) if use_latent_codes else None,
                #
                #     )
                #     target_ray_values = img_target
                #     coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                #     curr_loss, curr_fine_loss = 0.0, 0.0
                #     if rgb_fine is not None:
                #         curr_fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                #         curr_loss = curr_fine_loss
                #     else:
                #         curr_loss = coarse_loss
                #     loss += curr_loss + curr_fine_loss

                # 计算验证指标，生成验证日志
                loss /= len(i_val)
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_image("validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i)
                if rgb_fine:
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                    writer.add_image("validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i)
                writer.add_image("validation/img_target",cast_to_image(target_ray_values[..., :3]),i)
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


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

if __name__ == "__main__":
    main()
