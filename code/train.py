import argparse

import torch
import yaml
from yacs.config import CfgNode

import models
from dataset_utils import load_dataset, NeRFDataset
from nerf_helpers import seed_everything, get_embedding_function
from train_utils import Trainer, Mode


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="path to (.yml) config file.", default="../dataset/person_2/person_2_config.yml"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="path to load saved checkpoint.", default=""
    )
    parser.add_argument(
        "--savedir", type=str, help="Save images to this directory, if specified.", default="./renders/"
    )
    return parser


def load_data(cfg, device):
    # 加载数据集
    print("Starting data loading.")
    images, poses, expressions, bboxs, i_split, hwf, render_poses = load_dataset(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.testskip
    )
    dataset = NeRFDataset(Mode.TRAIN, cfg, device, images, poses, expressions, bboxs, i_split, hwf)
    # 设置背景
    trainable_background = False
    fixed_background = True
    dataset.load_background(trainable_background, fixed_background)
    # 设置潜在代码
    use_latent_codes = True
    dataset.load_latent_codes(use_latent_codes)
    # 生成射线重要性采样图
    p = 0.9
    dataset.generate_ray_importance_sampling_maps(p)
    print("Done data loading.")
    return dataset, trainable_background, fixed_background, use_latent_codes


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
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,

            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,

            latent_code_dim=cfg.models.fine.latent_code_dim
        )

    return model_coarse, model_fine, encode_position_fn, encode_direction_fn


def create_optimizer(cfg, model_coarse, model_fine, dataset, trainable_background, use_latent_codes):
    trainable_parameters = list(model_coarse.parameters())
    if model_fine:
        trainable_parameters += list(model_fine.parameters())
    if trainable_background:
        trainable_parameters.append(dataset.background)
    if use_latent_codes:
        trainable_parameters.append(dataset.latent_codes)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(trainable_parameters, lr=cfg.optimizer.lr)
    return optimizer


def main():
    # 设置调试时返回更多信息
    torch.autograd.set_detect_anomaly(True)

    # 命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # 加载其它基本配置
    seed_everything(cfg.experiment.randomseed)
    device = f"cuda:{cfg.experiment.device if cfg.experiment.device else 0}" if torch.cuda.is_available() else "cpu"

    # 加载数据
    dataset, trainable_background, fixed_background, use_latent_codes = load_data(cfg, device)

    # 设置NeRF模型
    print("Creating NeRF model.")
    model_coarse, model_fine, encode_position_fn, encode_direction_fn = create_nerf(cfg)

    # 设置优化器
    optimizer = create_optimizer(cfg, model_coarse, model_fine, dataset, trainable_background, use_latent_codes)

    # 设置训练器
    trainer = Trainer(Mode.TRAIN, args, cfg, device, model_coarse, model_fine, encode_position_fn, encode_direction_fn, optimizer, dataset)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
