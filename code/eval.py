import os
import time
from datetime import datetime

import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm
from yacs.config import CfgNode
from PIL import Image

from train_utils import run_one_iter_of_nerf
from nerf_helpers import create_parser, create_nerf, get_ray_bundle, cast_to_image
from load_dataset import load_data


def main():
    # 命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 读取配置文件 训练和测试的配置文件需要相同！！！！！！
    cfg = None
    with open(args.config, "r", encoding='utf-8') as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # 加载其它基本配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    print("Starting data loading.")
    images, poses, expressions, bboxs, i_split, hwf, render_360_poses = None, None, None, None, None, None, None
    H, W, focal = None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        images, poses, expressions, bboxs, i_split, hwf, render_360_poses = load_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            test=True
        )
        # 获取划分的训练集、验证集、测试集索引
        i_train, i_val, i_test = i_split
        # 获取相机内参
        H, W, focal = int(hwf[0]), int(hwf[1]), hwf[2]
        # 将图像从RGBA转为RGB，公式为targetRGB = sourceRGB * sourceA + BGcolor * (1 - sourceA)
        if cfg.nerf.train.white_background:  # BGcolor = 1
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    elif cfg.dataset.type.lower() == "llff":
        pass
    else:
        print("Load incorrect dataset.")
        return
    render_360_poses = render_360_poses.float().to(device)
    print("Done data loading.")

    # 设置NeRF模型
    model_coarse, model_fine, encode_position_fn, encode_direction_fn = create_nerf(cfg)
    model_coarse.to(device)
    if hasattr(cfg.models, "fine"):
        model_fine.to(device)

    # 加载检查点
    model_coarse, model_fine, background, latent_codes = None, None, None, None
    checkpoint = torch.load(args.checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
    if checkpoint["background"]:
        background = checkpoint["background"].to(device)
    if checkpoint["latent_codes"]:
        latent_codes = checkpoint["latent_codes"].to(device)
        # 加载idx_map，代表从“打乱后的图像顺序”映射到“原视频中的帧顺序”的图像索引
        idx_map = np.load(cfg.dataset.basedir + "/index_map.npy").astype(int)

    # 替换背景
    replace_background = True
    if replace_background:
        background = Image.open(os.path.join(cfg.dataset.basedir, 'bg', '00001.png'))
        background.thumbnail((H, W))
        background = torch.from_numpy(np.array(background).astype(float)).to(device)
        background = background/255

    # 生成训练日志
    os.makedirs(args.savedir, exist_ok=True)
    log_time = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    with open(os.path.join(args.savedir, log_time, "config.yml"), "w") as f:
        f.write(cfg.dump())
    with open(os.path.join(args.savedir, log_time, "args.txt"), "w") as f:
        for key in vars(args):
            f.write(f'{key}: {getattr(args, key)}\n')

    # 开启测试模式
    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    # 开始测试
    times_per_image = []
    render_poses = poses[i_test].float().to(device)
    render_expressions = expressions[i_test].float().to(device)
    for i in range(len(i_test)):
        start = time.time()

        with torch.no_grad():
            # 准备姿态
            pose = render_poses[i]
            pose = pose[:3, :4]
            # 准备表情
            expression = render_expressions[i]
            # 准备潜在代码（被固定住，或许可以修改一下）
            index_of_image_after_train_shuffle = idx_map[10,1]
            latent_code = latent_codes[index_of_image_after_train_shuffle].to(device)

            # 计算光线束的原点和方向
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose)
            ray_origins = ray_origins.view((-1, 3))
            ray_directions = ray_directions.view((-1, 3))

            # 重新渲染图像
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _, weights = run_one_iter_of_nerf(
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
                expressions=expression,
                background_prior=background.view(-1, 3) if (background is not None) else None,
                latent_code=latent_code,
                validation_image_shape=(H, W)
            )

            rgb = rgb_fine if rgb_fine is not None else rgb_coarse

        # 保存测试结果，生成测试日志
        times_per_image.append(time.time() - start)
        if args.savedir:
            savefile = os.path.join(args.savedir, log_time, "images", f"{i:04d}.png")
            imageio.imwrite(savefile, cast_to_image(rgb[..., :3]))
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    main()
