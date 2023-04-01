import os
import time
from datetime import datetime
from enum import Enum

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from torch_ema import ExponentialMovingAverage
from rich.console import Console

from nerf_helpers import mse2psnr, cast_to_image, img2mse


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Trainer(object):
    def __init__(self,
                 mode,
                 args,
                 cfg,
                 model,
                 optimizer,
                 dataset,
                 renderer
                 ):
        self.mode = mode
        self.args = args
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.renderer = renderer

        # 将模型加载到GPU
        model.to(self.dataset.device)
        renderer.to(self.dataset.device)
        # 创建日志文件夹
        self.logdir, self.writer, self.log_file = self.create_logdir()
        self.console = Console()
        # 加载检查点
        if os.path.exists(self.args.checkpoint):
            self.load_checkpoint()
        # 设置EMA
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg.trainer.ema_decay)
        # 设置AMP放大梯度的scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.trainer.fp16)

        # 设置训练参数
        self.start_iter = 0

    # ---------- utils ---------- #
    def create_logdir(self):
        # 根据模式(train/test)确定生成日志的基准文件夹
        basedir = self.cfg.experiment.logdir if self.mode is Mode.TRAIN else self.args.savedir
        # 创建时间戳
        time_stamp = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
        # 根据时间戳创建唯一命名的日志文件夹
        logdir = os.path.join(basedir, self.cfg.experiment.id, time_stamp)
        os.makedirs(logdir, exist_ok=True)
        # 开启tensorboard
        writer = SummaryWriter(logdir)
        # 保存当前运行程序的配置文件
        with open(os.path.join(basedir, self.cfg.experiment.id, time_stamp, "config.yml"), "w") as f:
            f.write(self.cfg.dump())
        # 保存当前运行程序的命令行
        with open(os.path.join(basedir, self.cfg.experiment.id, time_stamp, "args.txt"), "w") as f:
            for key in vars(self.args):
                f.write(f'{key}: {getattr(self.args, key)}\n')
        # 创建日志记录文件指针
        log_file = open(os.path.join(basedir, self.cfg.experiment.id, time_stamp, "log.txt"), "a+")
        return logdir, writer, log_file

    def __del__(self):
        self.log_file.close()

    def log(self, string):
        self.console.print(string)
        print(string, file=self.log_file)
        self.log_file.flush()

    def load_checkpoint(self):
        # 加载检查点
        checkpoint = torch.load(self.args.checkpoint)
        # 加载训练参数
        self.start_iter = checkpoint["iter"]
        self.renderer.mean_count = checkpoint['mean_count']
        self.renderer.mean_density = checkpoint['mean_density']
        # 加载模型参数
        self.model.load_state_dict(checkpoint["model"])
        self.renderer.load_state_dict(checkpoint["renderer"])
        if self.mode is Mode.TRAIN:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.ema.load_state_dict(checkpoint["ema"])
        if checkpoint["background"] is not None:
            self.dataset.background = torch.nn.Parameter(checkpoint['background'].to(self.dataset.device))
        if checkpoint["latent_codes"] is not None:
            self.dataset.latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(self.dataset.device))
            if self.mode is Mode.TEST:
                self.dataset.idx_map = np.load(self.cfg.dataset.basedir + "/index_map.npy").astype(int)

    def generate_train_log(self, i, loss, psnr, latent_code_loss):
        self.writer.add_scalar("train/loss", loss.item(), i)
        self.writer.add_scalar("train/psnr", psnr, i)
        if self.dataset.use_latent_codes:
            self.writer.add_scalar("train/code_loss", latent_code_loss.item(), i)

    def save_checkpoint(self, i, loss, psnr):
        checkpoint_dict = {
            "iter": i,
            "mean_count": self.renderer.mean_density,
            "mean_density": self.renderer.mean_count,
            # ---------- #
            "loss": loss,
            "psnr": psnr,
            # ---------- #
            "model": self.model.state_dict(),
            "renderer": self.renderer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict(),
            "background": self.dataset.background.data if self.dataset.fixed_background else None,
            "latent_codes": self.dataset.latent_codes.data if self.dataset.use_latent_codes else None
        }
        torch.save(
            checkpoint_dict,
            os.path.join(self.logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
        )
        tqdm.write("================== Saved Checkpoint =================")

    # ---------- train ---------- #
    def train(self):
        print("Marking untrained grid.")
        self.renderer.mark_untrained_grid()

        print("Starting loop.")
        for i in trange(self.start_iter + 1, self.cfg.experiment.train_iters + 1):
            # 开启训练模式
            self.model.train()
            self.renderer.train()
            # 训练阶段
            self.train_one_step(i)
            # 验证阶段
            if i % self.cfg.experiment.validate_every == 0:
                self.validate(i)
        print("Done!")

    def train_one_step(self, i):
        if i % self.cfg.renderer.update_extra_interval == 0:
            with torch.cuda.amp.autocast(enabled=self.cfg.trainer.fp16):
                self.renderer.update_extra_state(i)

        # 从训练集中选出一张图像及其表情、潜在代码、重要性采样图信息
        index = np.random.choice(self.dataset.i_train)
        image = self.dataset.images[index].to(self.dataset.device)
        expression = self.dataset.expressions[index].to(self.dataset.device)
        latent_code = self.dataset.latent_codes[index] if self.dataset.use_latent_codes else None
        ray_importance_sampling_map = self.dataset.ray_importance_sampling_maps[index].to(self.dataset.device)

        # 计算射线束的原点(N,3)和方向(N,3)（世界坐标系下）
        rays_o, rays_d, select_inds = self.renderer.get_rays(index, self.cfg.nerf.train.num_sample_rays, ray_importance_sampling_map)

        rgb_gt = torch.gather(image.view(-1, 3), dim=-2, index=torch.stack([select_inds, select_inds, select_inds], -1))
        background_gt = torch.gather(self.dataset.background.view(-1, 3), dim=-2, index=torch.stack([select_inds, select_inds, select_inds], -1)) if (
                self.dataset.trainable_background or self.dataset.fixed_background) else None

        # 计算渲染后的像素颜色
        with torch.cuda.amp.autocast(enabled=self.cfg.trainer.fp16):
            rgb_pred = self.renderer.render(
                self.model,
                self.mode,
                rays_o,  # 射线原点
                rays_d,  # 射线方向
                expression=expression,  # GT表情系数
                background_prior=background_gt,  # GT背景像素颜色
                latent_code=latent_code  # 潜在代码
            )

        # 计算损失函数
        loss = torch.nn.functional.mse_loss(rgb_pred, rgb_gt)
        latent_code_loss = torch.norm(latent_code) * 0.0005
        psnr = mse2psnr(loss.item())
        if self.dataset.use_latent_codes:
            loss = loss + latent_code_loss * 10

        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # 更新ema
        self.ema.update()

        # 学习率衰减
        num_decay_steps = self.cfg.scheduler.lr_decay * 1000
        lr_new = self.cfg.optimizer.lr * (self.cfg.scheduler.lr_decay_factor ** (i / num_decay_steps))
        # 0.1 ** (i / 250_000)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_new

        # 输出训练指标
        if i % self.cfg.experiment.print_every == 0 or i == self.cfg.experiment.train_iters:
            tqdm.write(
                f"[TRAIN] Iter: {str(i)} Loss: {str(loss.item())} PSNR: {str(psnr)} LatentReg: {str(latent_code_loss.item())}")

        # 生成训练日志
        self.generate_train_log(i, loss, psnr, latent_code_loss)

        # 保存检查点
        if i % self.cfg.experiment.save_every == 0 or i == self.cfg.experiment.train_iters:
            self.save_checkpoint(i, loss, psnr)

    # ---------- validate ---------- #
    def validate(self, i):
        self.model_coarse.eval()
        if self.model_fine:
            self.model_fine.eval()

        start = time.time()
        with torch.no_grad():
            rgb_coarse, rgb_fine = None, None
            target_ray_values = None

            loss = 0
            for img_idx in self.dataset.i_val:
                # 获取验证集图像及其位姿、表情、潜在代码信息
                img_target = self.dataset.images[img_idx].to(self.dataset.device)
                pose_target = self.dataset.poses[img_idx, :3, :4].to(self.dataset.device)
                expression_target = self.dataset.expressions[img_idx].to(self.dataset.device)
                latent_code = torch.zeros(32).to(self.dataset.device)

                # 计算穿过验证集图像所有像素的光线束的原点和方向（世界坐标系下）
                ray_origins, ray_directions = get_ray_bundle(self.dataset.H, self.dataset.W, self.dataset.intrinsics,
                                                             pose_target)
                ray_origins = ray_origins.view((-1, 3))
                ray_directions = ray_directions.view((-1, 3))

                # 计算渲染后的完整图像
                rgb_coarse, rgb_fine, weights = run_one_iter_of_nerf(
                    self.dataset.H,
                    self.dataset.W,
                    self.dataset.intrinsics,
                    self.model_coarse,
                    self.model_fine,
                    ray_origins,
                    ray_directions,
                    self.cfg,
                    mode="validation",
                    encode_position_fn=self.encode_position_fn,
                    encode_direction_fn=self.encode_direction_fn,
                    expressions=expression_target,
                    background_prior=self.dataset.background.view(-1, 3) if (
                            self.dataset.trainable_background or self.dataset.fixed_background) else None,
                    latent_code=latent_code if self.dataset.use_latent_codes else None,
                    validation_image_shape=(self.dataset.H, self.dataset.W, 3)
                )

                # 计算损失函数
                target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss = loss + coarse_loss
                if self.model_fine:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = loss + fine_loss

            # 计算验证指标，生成验证日志
            loss /= len(self.dataset.i_val)
            psnr = mse2psnr(loss.item())
            self.generate_validate_log(i, loss, psnr, coarse_loss, fine_loss, rgb_coarse, rgb_fine, target_ray_values,
                                       self.dataset.background, weights)

            tqdm.write(f"[VAL] Iter: {i} Loss: {str(loss.item())} PSNR: {str(psnr)} Time: {str(time.time() - start)}")

    def generate_validate_log(self, i, loss, psnr, coarse_loss, fine_loss, rgb_coarse, rgb_fine, target_ray_values,
                              background, weights):
        self.writer.add_scalar("validation/loss", loss.item(), i)
        self.writer.add_scalar("validation/psnr", psnr, i)
        self.writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
        self.writer.add_image("validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i)
        if self.model_fine:
            self.writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
            self.writer.add_image("validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i)
        self.writer.add_image("validation/img_target", cast_to_image(target_ray_values[..., :3]), i)
        if self.dataset.trainable_background or self.dataset.fixed_background:
            self.writer.add_image("validation/background", cast_to_image(background[..., :3]), i)
            self.writer.add_image("validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW')

    # ---------- test ---------- #
    def test(self):
        # 开启测试模式
        self.model_coarse.eval()
        if self.model_fine:
            self.model_fine.eval()

        # 开始测试
        times_per_image = []
        render_poses = self.dataset.poses[self.dataset.i_test].float().to(self.dataset.device)
        render_expressions = self.dataset.expressions[self.dataset.i_test].float().to(self.dataset.device)
        for i in tqdm(range(len(self.dataset.i_test))):
            start = time.time()

            with torch.no_grad():
                # 准备姿态
                pose = render_poses[i]
                pose = pose[:3, :4]
                # 准备表情
                expression = render_expressions[i]
                # 准备潜在代码（被固定住，或许可以修改一下）
                index_of_image_after_train_shuffle = self.dataset.idx_map[10, 1]
                latent_code = self.dataset.latent_codes[index_of_image_after_train_shuffle].to(self.dataset.device)

                # 计算光线束的原点和方向
                ray_origins, ray_directions = get_ray_bundle(self.dataset.H, self.dataset.W, self.dataset.intrinsics,
                                                             pose)
                ray_origins = ray_origins.view((-1, 3))
                ray_directions = ray_directions.view((-1, 3))

                # 重新渲染图像
                rgb_coarse, rgb_fine, weights = run_one_iter_of_nerf(
                    self.dataset.H,
                    self.dataset.W,
                    self.dataset.intrinsics,
                    self.model_coarse,
                    self.model_fine,
                    ray_origins,
                    ray_directions,
                    self.cfg,
                    mode="validation",
                    encode_position_fn=self.encode_position_fn,
                    encode_direction_fn=self.encode_direction_fn,
                    expressions=expression,
                    background_prior=self.dataset.background.view(-1, 3) if (
                                self.dataset.background is not None) else None,
                    latent_code=latent_code,
                    validation_image_shape=(self.dataset.H, self.dataset.W, 3)
                )

                rgb = rgb_fine if rgb_fine is not None else rgb_coarse

            # 保存测试结果，生成测试日志
            times_per_image.append(time.time() - start)
            savefile = os.path.join(self.logdir, "images", f"{i:04d}.png")
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            imageio.imwrite(savefile, rgb[..., :3])
