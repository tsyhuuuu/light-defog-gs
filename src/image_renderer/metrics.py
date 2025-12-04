from pathlib import Path
import os
import json
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser
from tqdm import tqdm

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    render_files = sorted([f for f in os.listdir(renders_dir) if not f.startswith(".")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if not f.startswith(".")])

    if len(render_files) != len(gt_files):
        raise ValueError(f"Folder size mismatch: {len(render_files)} vs {len(gt_files)}")

    for r, g in zip(render_files, gt_files):
        if r != g:
            raise ValueError(f"File mismatch: {r} vs {g}")

        render = Image.open(renders_dir / r)
        gt = Image.open(gt_dir / g)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(r)

    return renders, gts, image_names

def evaluate(renders_dir, gt_dir, save_dir=None, save_name="results.json"):
    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)

    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims, psnrs, lpipss = [], [], []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    mean_ssim = torch.tensor(ssims).mean().item()
    mean_psnr = torch.tensor(psnrs).mean().item()
    mean_lpips = torch.tensor(lpipss).mean().item()

    print("")
    print("  SSIM : {:>12.7f}".format(mean_ssim))
    print("  PSNR : {:>12.7f}".format(mean_psnr))
    print("  LPIPS: {:>12.7f}".format(mean_lpips))
    print("")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "SSIM": mean_ssim,
            "PSNR": mean_psnr,
            "LPIPS": mean_lpips
        }
        with open(save_dir / save_name, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_dir / save_name}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Compare two folders of images")
    parser.add_argument('--renders_dir', '-r', required=True, type=str, help="Folder with rendered images")
    parser.add_argument('--gt_dir', '-g', required=True, type=str, help="Folder with ground-truth images")
    parser.add_argument('--save_dir', '-sd', type=str, default=None, help="Directory to save results JSON")
    parser.add_argument('--save_name', '-sn', type=str, default="results.json", help="Name of the results JSON file")
    args = parser.parse_args()

    evaluate(args.renders_dir, args.gt_dir, args.save_dir, args.save_name)
