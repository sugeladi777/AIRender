import numpy as np
import os

import torch
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)

def cal_psnr(lightmap, lightmap_reconstruct, mask):
    mse = torch.mean((lightmap[:, :, mask >= 127] - lightmap_reconstruct[:, :, mask >= 127]) ** 2)
    max_value = torch.max(lightmap[:, :, mask >= 127])
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr.item()

def cal_ssim(lightmap, lightmap_reconstruct):
    with torch.no_grad():
        metric = StructuralSimilarityIndexMeasure(data_range=lightmap.max() - lightmap.min()).to(device)
        return metric(lightmap, lightmap_reconstruct).item()


def cal_lpips(lightmap, lightmap_reconstruct):    
    with torch.no_grad():
        lpips_value = lpips_fn(lightmap, lightmap_reconstruct).item()
        return lpips_value

def get_folder_size(folder_path):
    total_size = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                except OSError:
                    print(f"无法访问文件: {file_path}")
                    continue
    except OSError:
        print(f"无法访问文件夹: {folder_path}")
        return None
    
    return total_size

