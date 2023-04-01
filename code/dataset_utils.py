import os
import json

import imageio
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from code.nerf_helpers import meshgrid_xy
from train_utils import Mode


def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_dataset(
        basedir,  # 数据集的基准文件夹
        half_res=False,  # 是否以一半分辨率加载图像(half resolution)
        testskip=1,  # 测试集中每testskip张图像选取1张，避免测试集中相邻图像过于相似
        load_bbox=True,  # 是否加载边界框
        test=False  # 是否仅加载测试集
):
    # 从transforms_{train/val/test}.json中提取train/val/test的meta信息
    splits = ["train", "val", "test"] if not test else ['test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json'), 'r') as f:
            metas[s] = json.load(f)

    # 获取数据集的所有帧的数量
    num_frames = sum([len(meta['frames']) for meta in [metas[s] for s in splits]])
    # 从frame中获取图像、姿态、表情、边界框的形状
    sample_frame = metas['test']['frames'][0]
    img_shape = imageio.imread(os.path.join(basedir, sample_frame['file_path'] + '.png')).shape
    pose_shape = np.array(sample_frame['transform_matrix']).shape
    expression_shape = np.array(sample_frame['expression']).shape
    bbox_shape = np.array(sample_frame['bbox']).shape if 'bbox' in sample_frame.keys() else (4,)
    # 图像、姿态、表情、边界框初始化，初始化类型为np.float32类型
    imgs = np.zeros(tuple([num_frames] + list(img_shape)), dtype=np.float32)
    poses = np.zeros(tuple([num_frames] + list(pose_shape)), dtype=np.float32)
    expressions = np.zeros(tuple([num_frames] + list(expression_shape)), dtype=np.float32)
    bboxs = np.zeros(tuple([num_frames] + list(bbox_shape)), dtype=np.float32)
    # 将meta信息拆分为图像、姿态、表情、边界框
    idx = 0
    counts = [0]
    for s in splits:
        print(f'{s} dataset is loading.')
        meta = metas[s]
        skip = 1 if s == "train" else (testskip if testskip != 0 else 1)

        # 获取图像、姿态、表情、边界框
        for frame in tqdm(meta['frames'][::skip]):
            # 图像
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs[idx, ...] = imageio.imread(fname) / 255.0
            # 姿态
            poses[idx, ...] = np.array(frame['transform_matrix'])
            # 表情
            expressions[idx, ...] = np.array(frame['expression'])
            # 边界框
            if load_bbox:
                bboxs[idx, ...] = np.array(frame['bbox'] if 'bbox' in frame.keys() else np.array([0.0, 1.0, 0.0, 1.0]))
            idx += 1

        # 对当前的训练集/验证集/测试集进行计数
        counts.append(idx)

    # 生成训练集/验证集/测试集的样本索引
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    # 计算相机的内参
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['test']['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = np.array(metas['test']["intrinsics"]) if metas['test']["intrinsics"] else np.array(
        [focal, focal, 0.5 * W, 0.5 * H])

    # 直接生成渲染图像时应当使用的人脸姿态
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0
    )

    # 将numpy转化为torch
    # 转化图像：节省性能时，每张图像只加载原先的1/4
    if half_res:
        H, W = H // 2, W // 2
        intrinsics = intrinsics * 0.5
        imgs = [
            torch.from_numpy(cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA))
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
    else:
        imgs = [
            torch.from_numpy(imgs[i])
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
    # 转化姿态
    poses = torch.from_numpy(poses)
    # 转化表情
    expressions = torch.from_numpy(expressions)
    # 转化边界框
    bboxs[:, 0:2] *= H
    bboxs[:, 2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()

    # 返回最终值
    return imgs, poses, expressions, bboxs, i_split, H, W, intrinsics, render_poses


class NeRFDataset(object):
    def __init__(self, mode, cfg, device, images, poses, expressions, bboxs, i_split, H, W, intrinsics):
        self.mode = mode
        self.cfg = cfg
        self.device = device
        self.images = images
        self.poses = poses
        self.expressions = expressions
        self.bboxs = bboxs
        self.trainable_background, self.fixed_background, self.background = None, None, None
        self.use_latent_codes, self.latent_codes, self.idx_map = None, None, None
        self.p, self.ray_importance_sampling_maps = None, None
        self.num_data = self.images.shape[0]

        # 获取划分的训练集、验证集、测试集索引
        if self.mode is Mode.TRAIN:
            self.i_train, self.i_val, self.i_test = i_split
        else:
            self.i_train, self.i_val, self.i_test = None, None, i_split[0]
        # 获取相机内参
        self.H = int(H)
        self.W = int(W)
        self.intrinsics = intrinsics
        # 将图像从RGBA转为RGB，公式为targetRGB = sourceRGB * sourceA + BGcolor * (1 - sourceA)
        if cfg.nerf.train.white_background:  # BGcolor = 1
            self.images = self.images[..., :3] * self.images[..., -1:] + (1.0 - self.images[..., -1:])

    # ---------- train ---------- #
    def load_background(self, trainable_background, fixed_background):
        self.trainable_background = trainable_background
        self.fixed_background = fixed_background
        if trainable_background:  # 可训练背景
            print("Creating trainable background.")
            # 背景初始化为训练集图像的平均值
            with torch.no_grad():
                avg_img = torch.mean(self.images[self.i_train], axis=0)
                # ??? 模糊背景图
                # ------------
                background = torch.tensor(avg_img, device=self.device)
            background.requires_grad = True
        elif fixed_background:  # 固定的真实背景
            print("Loading GT background.")
            background = Image.open(os.path.join(self.cfg.dataset.basedir, 'bg', '00001.png'))
            background.thumbnail((self.H, self.W))
            background = torch.from_numpy(np.array(background).astype(np.float32)).to(self.device)
            background = background / 255
        else:
            background = None
        self.background = background

    def load_latent_codes(self, use_latent_codes):
        self.use_latent_codes = use_latent_codes
        if use_latent_codes:
            print("Setting latent codes.")
            latent_codes = torch.zeros((len(self.i_train), 32), device=self.device)
            latent_codes.requires_grad = True
        else:
            latent_codes = None
        self.latent_codes = latent_codes

    def generate_ray_importance_sampling_maps(self, p):
        self.p = p
        ray_importance_sampling_maps = []
        for i in self.i_train:
            # 提取当前图像的边界框
            bbox = self.bboxs[i]
            # 边界框内的像素概率设置为p，边界框外的像素概率设置为1-p
            probs = np.zeros((self.H, self.W))
            probs.fill(1 - p)
            probs[bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
            # 概率归一化
            probs = (1 / probs.sum()) * probs
            # 压扁后加入射线重要性采样图中
            ray_importance_sampling_maps.append(torch.from_numpy(probs.reshape(-1)))
        self.ray_importance_sampling_maps = ray_importance_sampling_maps

    # ---------- test ---------- #
    def replace_background(self, img_path):
        background = Image.open(img_path)
        background.thumbnail((self.H, self.W))
        background = torch.from_numpy(np.array(background).astype(float)).to(self.device)
        background = background / 255
        self.background = background
