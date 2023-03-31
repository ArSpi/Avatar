import argparse

import torch
import yaml
from yacs.config import CfgNode

import models
from renderer import NeRFRenderer
from dataset_utils import load_dataset, NeRFDataset
from nerf_helpers import seed_everything, get_embedding_function
from train_utils import Trainer, Mode


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="path to (.yml) config file.", default="../dataset/person_1/person_1_config.yml"
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
    images, poses, expressions, bboxs, i_split, H, W, intrinsics, render_poses = load_dataset(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.testskip
    )
    dataset = NeRFDataset(Mode.TRAIN, cfg, device, images, poses, expressions, bboxs, i_split, H, W, intrinsics)
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


def create_model(cfg):
    model = models.HashGridNetwork(
        # 模型结构
        num_layers=cfg.models.num_layers,  # 模型的层数
        hidden_dim=cfg.models.hidden_dim,  # 隐藏层的大小
        geo_feat_dim=cfg.models.geo_feat_dim,
        num_layers_color=cfg.models.num_layers_color,
        hidden_dim_color=cfg.models.hidden_dim_color,
        #
        bound=cfg.bound,
        # 潜在代码
        expression_dim=cfg.models.expression_dim,
        latent_code_dim=cfg.models.latent_code_dim
    )

    return model


def create_renderer(cfg, dataset):
    renderer = NeRFRenderer(
        cfg=cfg,
        dataset=dataset,
        bound=cfg.bound,
        density_scale=1,
        min_near=cfg.min_near,
        density_thresh=cfg.density_thresh
    )

    return renderer


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
    # model_coarse, model_fine, encode_position_fn, encode_direction_fn = create_model(cfg)
    model = create_model(cfg)

    # 设置优化器
    optimizer = torch.optim.Adam(model.get_params(dataset, cfg.optimizer.lr, trainable_background, use_latent_codes))

    # 设置渲染器
    renderer = create_renderer(cfg, dataset)

    # 设置训练器
    trainer = Trainer(Mode.TRAIN, args, cfg, device, model, optimizer, dataset, renderer)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
