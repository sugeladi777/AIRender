import numpy as np
import os

import torch
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Avoid creating device-bound metric objects at import time. Instead create or
# move them to the device of the tensors at call time. Cache LPIPS per-device
# to avoid repeated heavy re-initialization.
_lpips_cache = {}


def cal_psnr(lightmap, lightmap_reconstruct, mask):
    # Ensure boolean mask on same device as inputs
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=lightmap.device)
    else:
        mask = mask.to(lightmap.device)

    mask_bool = mask >= 127
    # If no valid pixels, return NaN
    if not torch.any(mask_bool):
        return float('nan')

    gt = lightmap[:, :, mask_bool]
    rec = lightmap_reconstruct[:, :, mask_bool]
    mse = torch.mean((gt - rec) ** 2)
    if mse == 0:
        return float('inf')
    max_value = torch.max(gt)
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return float(psnr.item())


def cal_ssim(lightmap, lightmap_reconstruct):
    with torch.no_grad():
        device = lightmap.device
        # data_range expects a scalar; compute on CPU-safe scalar
        data_range = float((lightmap.max() - lightmap.min()).detach().cpu().item())
        metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        return float(metric(lightmap.to(device), lightmap_reconstruct.to(device)).item())


def cal_lpips(lightmap, lightmap_reconstruct):
    with torch.no_grad():
        device = lightmap.device
        key = str(device)
        if key not in _lpips_cache:
            # initialize LPIPS on the same device as the inputs
            _lpips_cache[key] = lpips.LPIPS(net='alex').to(device)
        lpips_fn = _lpips_cache[key]
        val = lpips_fn(lightmap.to(device), lightmap_reconstruct.to(device)).item()
        return float(val)

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

