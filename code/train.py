import argparse
import torch
import yaml
import numpy as np
from yacs.config import CfgNode

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
    hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
    # 将图像从RGBA转为RGB，公式为targetRGB = sourceRGB * sourceA + BGcolor * (1 - sourceA)
    if cfg.nerf.train.white_background:  # BGcolor = 1
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    print("Done data loading.")

    # 加载其它基本配置
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 设置NeRF模型
    model_coarse, model_fine = create_nerf(cfg)
    model_coarse.to(device)
    if hasattr(cfg.models, "fine"):
        model_fine.to(device)


if __name__ == "__main__":
    main()
