import torch
import yaml
from yacs.config import CfgNode

from dataset_utils import load_data, NeRFDataset
from nerf_helpers import create_parser, create_nerf, seed_everything
from train_utils import Trainer, Mode


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
    seed_everything(cfg.experiment.randomseed)
    device = f"cuda:{cfg.experiment.device if cfg.experiment.device else 0}" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    print("Starting data loading.")
    images, poses, expressions, bboxs, i_split, hwf, render_poses = load_data(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.testskip
    )
    dataset = NeRFDataset(cfg, device, images, poses, expressions, bboxs, i_split, hwf)
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

    # 设置NeRF模型
    print("Creating NeRF model.")
    model_coarse, model_fine, encode_position_fn, encode_direction_fn = create_nerf(cfg)

    # 设置优化器
    trainable_parameters = list(model_coarse.parameters())
    if model_fine:
        trainable_parameters += list(model_fine.parameters())
    if trainable_background:
        trainable_parameters.append(dataset.background)
    if use_latent_codes:
        trainable_parameters.append(dataset.latent_codes)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(trainable_parameters, lr=cfg.optimizer.lr)

    # 设置训练器
    trainer = Trainer(Mode.TRAIN, args, cfg, device, model_coarse, model_fine, encode_position_fn, encode_direction_fn, optimizer, dataset)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
