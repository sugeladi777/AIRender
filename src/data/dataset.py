from __future__ import annotations
"""
HPRC 数据集读取与采样（比赛数据）

功能要点：
- 从 Data_HPRC/config.json 读取 lightmap 列表与分辨率；
- 支持按 lightmap 索引选择一张贴图，遍历/随机采样多个时刻（含 5.9/18.1 特殊时刻，键为 590/1810）；
- 读取 .bin_0 原始二进制：lightmap float32、mask int8；
- 采样时使用 mask>=127 作为有效像素（可关闭 use_masks），返回 (xy, t_feat, rgb)；
- 可选 residual_mode：以 baseline_time（默认 12）为基准，学习残差 rgb - baseline；
- 与现有 LightMapTimeDataset 接口一致（xy∈[-1,1]，时间特征为 Fourier 编码）。

使用示例：
    ds = HPRCLightmapDataset(hprc_dir='Data_HPRC', lightmap_index=0,
                             time_harmonics=8, sample_mode='random', samples_per_epoch=1_000_000,
                             residual_mode=True, baseline_time=12)
    xy, t_feat, rgb = ds[0]
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.encoding import encode_time


def _ensure_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    return path


def _load_config(hprc_dir: str) -> Dict:
    cfg_path = os.path.join(hprc_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found under {hprc_dir}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _sorted_time_keys(lightmaps: Dict[str, str]) -> List[int]:
    """
    输入为 {'0': 'xxx', '100': 'xxx', '590': 'xxx', ...}；输出排序后的 int 键列表，例如 [0, 100, 200, ..., 1810, ...]
    """
    try:
        keys = [int(k) for k in lightmaps.keys()]
    except Exception:
        keys = []
        for k in lightmaps.keys():
            try:
                keys.append(int(k))
            except Exception:
                continue
    keys.sort()
    return keys


def _read_lightmap(path: str, H: int, W: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    expected = H * W * 3
    if data.size != expected:
        raise ValueError(f"lightmap size mismatch: got {data.size}, expect {expected} (H={H}, W={W}) at {path}")
    return data.reshape(H, W, 3)


def _read_mask(path: str, H: int, W: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int8)
    expected = H * W
    if data.size != expected:
        raise ValueError(f"mask size mismatch: got {data.size}, expect {expected} (H={H}, W={W}) at {path}")
    return data.reshape(H, W)


@dataclass
class HPRCLightmapSpec:
    level: str
    id: int
    resolution: Tuple[int, int]  # (H, W)
    time_keys: List[int]         # e.g. [0,100,...,1810,...]
    lightmaps: Dict[int, str]    # time_key -> filename
    masks: Dict[int, str]        # time_key -> filename


def load_hprc_lightmap_spec(hprc_dir: str, lightmap_index: int) -> HPRCLightmapSpec:
    cfg = _load_config(hprc_dir)
    lst = cfg.get('lightmap_list', [])
    if not (0 <= lightmap_index < len(lst)):
        raise IndexError(f"lightmap_index {lightmap_index} out of range [0, {len(lst)-1}]")
    item = lst[lightmap_index]

    level = item['level']
    lid = int(item['id'])
    res = item['resolution']
    H = int(res['height'])
    W = int(res['width'])
    lm_dict = item['lightmaps']
    mk_dict = item['masks']
    time_keys = _sorted_time_keys(lm_dict)
    # 统一映射为 int->str
    lightmaps = {int(k): v for k, v in lm_dict.items()}
    masks = {int(k): v for k, v in mk_dict.items()}

    return HPRCLightmapSpec(
        level=level,
        id=lid,
        resolution=(H, W),
        time_keys=time_keys,
        lightmaps=lightmaps,
        masks=masks,
    )


class HPRCLightmapDataset(Dataset):
    """
    读取 HPRC 比赛数据，按像素返回 (xy_norm, t_feat, rgb)。

    - xy_norm: [-1,1] 两维坐标
    - t_feat: 时间 Fourier 编码（encode_time，K=time_harmonics）
    - rgb: float32 真实值或相对于 baseline 的残差（residual_mode=True）

    数据根目录期望结构：
    Data_HPRC/
      ├─ config.json
      └─ Data/
          ├─ lightmapRawData_*.bin_0
          └─ lightmapCoverage_*.bin_0
    """

    def __init__(
        self,
        hprc_dir: str,
        lightmap_index: int = 0,
        time_harmonics: int = 8,
        sample_mode: str = 'random',
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
        residual_mode: bool = False,
        baseline_time: int = 12,
        use_masks: bool = True,
    ) -> None:
        super().__init__()
        self.hprc_dir = _ensure_dir(hprc_dir)
        self.data_dir = os.path.join(self.hprc_dir, 'Data')
        if not os.path.isdir(self.data_dir):
            # 兼容：有些数据可能直接放在 hprc_dir 根目录
            self.data_dir = self.hprc_dir

        self.spec = load_hprc_lightmap_spec(self.hprc_dir, lightmap_index)
        self.time_harmonics = int(time_harmonics)
        self.sample_mode = str(sample_mode)
        self.rng = np.random.default_rng(seed)
        self.use_masks = bool(use_masks)

        # 残差与 baseline
        self.residual_mode = bool(residual_mode)
        try:
            self.baseline_key = int(round(float(baseline_time) * 100))
        except Exception:
            self.baseline_key = 1200
        if self.baseline_key not in self.spec.time_keys:
            # 回退：若不在键中（例如只提供整点），则选最接近的整点
            nearest = min(self.spec.time_keys, key=lambda k: abs(k - self.baseline_key))
            self.baseline_key = nearest

        self.H, self.W = self.spec.resolution

        # 惰性缓存：每个时间点的有效像素索引，以及 baseline 帧
        self._valid_indices: Dict[int, np.ndarray] = {}
        self._baseline_img: Optional[np.ndarray] = None  # [H,W,3] in float32

        total = len(self.spec.time_keys) * self.H * self.W
        if self.sample_mode == 'all':
            self.length = total
        else:
            self.length = samples_per_epoch if samples_per_epoch is not None else min(total, 1_000_000)

    # ------------- 公共属性 -------------
    @property
    def size(self) -> Tuple[int, int]:
        return self.H, self.W

    @property
    def baseline_image(self) -> Optional[np.ndarray]:
        if not self.residual_mode:
            return None
        if self._baseline_img is None:
            path = os.path.join(self.data_dir, self.spec.lightmaps[self.baseline_key])
            self._baseline_img = _read_lightmap(path, self.H, self.W)
        return self._baseline_img

    # ------------- 内部工具 -------------
    def _get_valid_indices(self, t_key: int) -> np.ndarray:
        """
        返回 mask 有效像素的平面索引数组（形如 [N,2]，列为 [y,x]）。在首次访问该时刻时构建并缓存。
        若 use_masks=False，则返回全图像素索引。
        """
        if not self.use_masks:
            ys, xs = np.meshgrid(np.arange(self.H, dtype=np.int32), np.arange(self.W, dtype=np.int32), indexing='ij')
            coords = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)
            return coords

        if t_key in self._valid_indices:
            return self._valid_indices[t_key]

        mk_name = self.spec.masks[t_key]
        mk_path = os.path.join(self.data_dir, mk_name)
        mk = _read_mask(mk_path, self.H, self.W)
        valid = mk >= 127
        ys, xs = np.where(valid)
        coords = np.stack([ys.astype(np.int32), xs.astype(np.int32)], axis=-1)
        if coords.size == 0:
            # 极端情况下无有效像素，则退化为全像素
            ys, xs = np.meshgrid(np.arange(self.H, dtype=np.int32), np.arange(self.W, dtype=np.int32), indexing='ij')
            coords = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)
        self._valid_indices[t_key] = coords
        return coords

    def _sample_random(self) -> Tuple[int, int, int]:
        # 随机挑选时刻、有效像素坐标 (t_key, y, x)
        t_key = int(self.rng.choice(self.spec.time_keys))
        coords = self._get_valid_indices(t_key)
        idx = int(self.rng.integers(0, coords.shape[0]))
        y, x = int(coords[idx, 0]), int(coords[idx, 1])
        return t_key, y, x

    # ------------- Dataset 接口 -------------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.sample_mode == 'all':
            T = len(self.spec.time_keys)
            t_idx = idx // (self.H * self.W)
            rem = idx % (self.H * self.W)
            y = rem // self.W
            x = rem % self.W
            t_key = self.spec.time_keys[t_idx % T]
            # 若启用 mask 且该像素无效，则回退为随机一个有效像素
            if self.use_masks:
                coords = self._get_valid_indices(t_key)
                if coords.shape[0] == self.H * self.W:
                    pass
                else:
                    # 检查 (y,x) 是否有效
                    # 这里简单起见，若无效则随机替换
                    mk_name = self.spec.masks[t_key]
                    mk_path = os.path.join(self.data_dir, mk_name)
                    mk = _read_mask(mk_path, self.H, self.W)
                    if mk[y, x] < 127:
                        tid, y, x = self._sample_random()
                        t_key = tid
        else:
            t_key, y, x = self._sample_random()

        # 读取该时刻 lightmap，并取像素
        lm_name = self.spec.lightmaps[t_key]
        lm_path = os.path.join(self.data_dir, lm_name)
        lm = _read_lightmap(lm_path, self.H, self.W)
        rgb = lm[y, x, :].astype(np.float32)

        if self.residual_mode:
            base = self.baseline_image
            rgb = rgb - base[y, x, :].astype(np.float32)

        # 归一化坐标到 [-1,1]
        x_norm = (x + 0.5) / self.W * 2 - 1
        y_norm = (y + 0.5) / self.H * 2 - 1
        xy = np.array([x_norm, y_norm], dtype=np.float32)

        # 时间编码：将 t_key 从 [0..2300] 映射到 [0,24] 再归一化为 [0,1]
        t_norm = (t_key / 100.0) / 24.0
        t_feat = encode_time(t_norm, K=self.time_harmonics)

        return (
            torch.from_numpy(xy),           # [2]
            torch.from_numpy(t_feat),       # [2K]
            torch.from_numpy(rgb),          # [3]
        )


__all__ = [
    'HPRCLightmapDataset',
    'HPRCLightmapSpec',
    'load_hprc_lightmap_spec',
]

