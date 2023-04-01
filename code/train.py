import argparse

import torch
import yaml
from yacs.config import CfgNode

import models
from renderer import NeRFRenderer
from dataset_utils import load_dataset, NeRFDataset
from nerf_helpers import seed_everything
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
    model = models.HashGridNetwork(cfg)

    # 设置优化器
    optimizer = torch.optim.Adam(model.get_params(dataset, cfg.optimizer.lr, trainable_background, use_latent_codes))

    # 设置渲染器
    renderer = NeRFRenderer(cfg, model, dataset)

    # 设置训练器
    trainer = Trainer(Mode.TRAIN, args, cfg, model, optimizer, dataset, renderer)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
