from __future__ import annotations
import os
import glob
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..utils.time_encoding import encode_time


def _load_images(image_dir: str) -> np.ndarray:
    """从目录加载图像序列并返回形状为 [T, H, W, 3] 的 numpy 数组。

    要点：按字典序排序文件以保证时间顺序一致；期望 24 帧，否则会打印警告。
    """
    exts = ("*.png", "*.jpg", "*.jpeg")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(image_dir, e)))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if len(files) != 24:
        # 非严格错误，仅提示：时间索引仍按排序顺序计算
        print(f"[WARN] Expected 24 images, found {len(files)}. Will proceed but time indexing assumes sorted order.")

    imgs = []
    for f in files:
        im = Image.open(f).convert('RGB')
        arr = np.array(im, dtype=np.float32) / 255.0  # H,W,3 in [0,1]
        imgs.append(arr)
    # 确认所有图像尺寸一致
    H, W = imgs[0].shape[:2]
    for i, arr in enumerate(imgs):
        if arr.shape[:2] != (H, W):
            raise ValueError(f"Image size mismatch at {i}: {arr.shape[:2]} != {(H, W)}")
    stack = np.stack(imgs, axis=0)  # [T, H, W, 3]
    return stack


class LightMapTimeDataset(Dataset):
    """
    数据集：按像素返回 (xy_norm, t_feat, rgb)

    - xy_norm: 归一化坐标 [-1,1]
    - t_feat: 时间特征（Fourier 编码）
    - rgb: 目标颜色

    支持两种采样模式：'all'（遍历全部像素）或 'random'（每次随机采样）。
    """
    def __init__(self,
                 image_dir: str,
                 time_harmonics: int = 2,
                 sample_mode: str = 'random',
                 samples_per_epoch: Optional[int] = None,
                 seed: int = 42):
        super().__init__()
        self.stack = _load_images(image_dir)  # [T, H, W, 3]
        self.T, self.H, self.W, _ = self.stack.shape
        self.time_harmonics = time_harmonics
        self.sample_mode = sample_mode
        self.rng = np.random.default_rng(seed)

        total = self.T * self.H * self.W
        if self.sample_mode == 'all':
            self.length = total
        else:
            # 随机采样时默认上限为 1e6，避免一次 epoch 过大
            self.length = samples_per_epoch if samples_per_epoch is not None else min(total, 1_000_000)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # 依据采样模式返回 (xy, t_feat, rgb)
        if self.sample_mode == 'all':
            # 以 t,y,x 的顺序枚举所有像素
            t = idx // (self.H * self.W)
            rem = idx % (self.H * self.W)
            y = rem // self.W
            x = rem % self.W
        else:
            # 随机采样单个像素帧
            t = self.rng.integers(0, self.T)
            y = self.rng.integers(0, self.H)
            x = self.rng.integers(0, self.W)

        # 坐标归一化到 [-1,1]
        x_norm = (x + 0.5) / self.W * 2 - 1
        y_norm = (y + 0.5) / self.H * 2 - 1
        xy = np.array([x_norm, y_norm], dtype=np.float32)

        # 时间编码（t_norm 在 [0,1]）
        t_norm = t / 24.0  # 即使 T != 24，仍按 24 小时周期归一化
        t_feat = encode_time(t_norm, self.time_harmonics)  # [2*K]

        rgb = self.stack[t, y, x, :].astype(np.float32)

        return (
            torch.from_numpy(xy),
            torch.from_numpy(t_feat),
            torch.from_numpy(rgb),
        )

    @property
    def size(self) -> Tuple[int, int]:
        return self.H, self.W
