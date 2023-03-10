import argparse
import glob
import os
from datetime import datetime

import PIL
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm


class Metric(object):
    def __init__(self):
        self.value = 0.0
        self.num = 0

    def reset(self):
        self.value, self.num = 0.0, 0.0

    def update(self, x):
        self.value += x
        self.num += 1

    def cal(self):
        assert self.num != 0, "No Metric!"
        return self.value / self.num


def ssim_loss(im1, im2):
    return ssim(im1, im2, channel_axis=2)


def psnr_loss(im1, im2):
    return psnr(im1, im2)


def lpips_loss(im1, im2):
    lpips_fn = lpips.LPIPS(net='alex')
    with torch.no_grad():
        loss = lpips_fn(
            torch.FloatTensor(im1.astype('float32')).permute(2, 0, 1).unsqueeze(0),
            torch.FloatTensor(im2.astype('float32')).permute(2, 0, 1).unsqueeze(0)
        ).cpu()
    return loss.item()


def folder_mode(args):
    gt_images_path = glob.glob(os.path.join(args.gt_path, '*g'))
    syn_images_path = glob.glob(os.path.join(args.syn_path, '*g'))

    assert len(gt_images_path) == len(syn_images_path), 'Different number between GT images and synthetic images.'

    os.makedirs(args.output_dir, exist_ok=True)
    time = datetime.now().strftime('%Y-%m-%d %H_%M_%S')

    with open(f"metrics_{time}.txt", "w") as f:
        f.write(f'folders metrics comparison.\n')
        f.write(f'path to GT images: {args.gt_path}.\n')
        f.write(f'path to synthetic images: {args.syn_path}.\n')
        f.write('\n')
        f.write('='*80)
        f.write('\n')
        print(f'folders metrics comparison.\n')
        print(f'path to GT images: {args.gt_path}.\n')
        print(f'path to synthetic images: {args.syn_path}.\n')
        print('\n')
        print('=' * 80)
        print('\n')

        SSIM, PSNR, LPIPS = Metric(), Metric(), Metric()
        num_images = len(gt_images_path)

        for i in tqdm(range(num_images)):
            gt_image = np.array(PIL.Image.open(os.path.join(args.gt_path, gt_images_path[i]))) / 255
            syn_image = np.array(PIL.Image.open(os.path.join(args.syn_path, syn_images_path[i]))) / 255

            assert gt_image.shape == syn_image.shape, f'Different shapes between GT image {gt_images_path[i]} and synthetic image {syn_images_path[i]}.'

            curr_ssim_loss = ssim_loss(gt_image, syn_image)
            curr_psnr_loss = psnr_loss(gt_image, syn_image)
            curr_lpips_loss = lpips_loss(gt_image, syn_image)

            SSIM.update(curr_ssim_loss)
            PSNR.update(curr_psnr_loss)
            LPIPS.update(curr_lpips_loss)

            f.write(f'image {i}   ssim {curr_ssim_loss}   psnr {curr_psnr_loss}   lpips {curr_lpips_loss}\n')
            print(f'image {i}   ssim {curr_ssim_loss}   psnr {curr_psnr_loss}   lpips {curr_lpips_loss}\n')

        f.write('\n')
        f.write('=' * 80)
        f.write('\n')
        f.write('Summary\n')
        f.write('-' * 80)
        f.write('\n   mean SSIM:\t%5f' % (SSIM.cal()))
        f.write('\n   mean PSNR:\t%5f' % (PSNR.cal()))
        f.write('\n   mean LPIPS:\t%5f' % (LPIPS.cal()))
        print('\n')
        print('=' * 80)
        print('\n')
        print('Summary\n')
        print('-' * 80)
        print('\n   mean SSIM:\t%5f' % (SSIM.cal()))
        print('\n   mean PSNR:\t%5f' % (PSNR.cal()))
        print('\n   mean LPIPS:\t%5f' % (LPIPS.cal()))


def image_mode(args):
    gt_image = np.array(PIL.Image.open(args.gt_path)) / 255
    syn_image = np.array(PIL.Image.open(args.syn_path)) / 255

    assert gt_image.shape == syn_image.shape, f'Different shapes between two images.'

    curr_ssim_loss = ssim_loss(gt_image, syn_image)
    curr_psnr_loss = psnr_loss(gt_image, syn_image)
    curr_lpips_loss = lpips_loss(gt_image, syn_image)

    print(f'ssim {curr_ssim_loss}   psnr {curr_psnr_loss}   lpips {curr_lpips_loss}\n')


def main():
    default_gt_path = ""
    default_syn_path = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default=default_gt_path, help="path to GT images")
    parser.add_argument("--syn_path", type=str, default=default_syn_path, help="path to synthetic images")
    parser.add_argument("--mode", type=str, default="folder", help="calculating metrics with folders or images")
    parser.add_argument("--output_dir", type=str, default="./metrics_log")
    args = parser.parse_args()

    folder_mode(args) if args.mode == "folder" else image_mode(args)

if __name__ == "__main__":
    main()