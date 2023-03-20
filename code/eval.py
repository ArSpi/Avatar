import os

import torch
import yaml
from yacs.config import CfgNode

from train_utils import Trainer
from nerf_helpers import create_parser, create_nerf, seed_everything
from dataset_utils import load_data, NeRFDataset


def main():
    # 命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 读取配置文件 训练和测试的配置文件需要相同！！！！！！
    with open(args.config, "r", encoding='utf-8') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # 加载其它基本配置
    seed_everything(cfg.experiment.randomseed)
    device = f"cuda:{cfg.experiment.device if cfg.experiment.device else 0}" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    print("Starting data loading.")
    images, poses, expressions, bboxs, i_split, hwf, render_360_poses = load_data(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        testskip=cfg.dataset.testskip,
        test=True
    )
    dataset = NeRFDataset(cfg, device, images, poses, expressions, bboxs, i_split, hwf)
    print("Done data loading.")

    # 设置NeRF模型
    model_coarse, model_fine, encode_position_fn, encode_direction_fn = create_nerf(cfg)

    # 设置训练器
    trainer = Trainer(args, cfg, device, model_coarse, model_fine, encode_position_fn, encode_direction_fn, None, dataset)

    # 替换背景
    replace_background = True
    if replace_background:
        dataset.replace_background(os.path.join(cfg.dataset.basedir, 'bg', '00001.png'))

    # 测试过程，生成重建的图像
    trainer.test()

if __name__ == "__main__":
    main()
