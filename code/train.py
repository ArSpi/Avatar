import argparse
import os
from datetime import datetime

import torch
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from yacs.config import CfgNode
from PIL import Image

from load_dataset import load_data
import models
from nerf_helpers import get_embedding_function


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
    ) if cfg.models.coarse.use_viewdirs else None

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






    print("Done!")

if __name__ == "__main__":
    main()
