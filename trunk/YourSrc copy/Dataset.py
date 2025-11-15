"""
Dataset.py: 光照贴图数据加载和预处理模块

负责加载光照贴图数据、计算残差、准备训练坐标和分段标记。
支持白天/夜晚模型的数据分离和预处理。
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


class LightmapDataLoader:
    """
    光照贴图数据加载器

    处理光照贴图数据的加载、预处理和张量准备，包括残差计算和昼夜分段。
    """

    def __init__(self, dataset_path: str):
        """
        初始化数据加载器

        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = dataset_path
        self.config = self.load_dataset_config()

    def load_dataset_config(self, config_file: str = "config.json") -> Dict:
        """
        加载数据集配置文件

        Args:
            config_file: 配置文件名

        Returns:
            配置字典
        """
        config_path = os.path.join(self.dataset_path, config_file)
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_lightmap_and_mask(
        self,
        lightmap: Dict,
        times: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载单个光照贴图在多个时间点的光照图和遮罩

        Args:
            lightmap: 光照贴图信息字典
            times: 时间点列表

        Returns:
            光照贴图数据和遮罩数据
        """
        lightmap_names = lightmap["lightmaps"]
        mask_names = lightmap["masks"]
        resolution = lightmap["resolution"]

        H, W = resolution["height"], resolution["width"]
        # 读取 lightmap 序列并展平为 (H*W,3)，随后在时间维拼接
        lightmap_in_different_time: List[np.ndarray] = []
        for t_key in times:
            lightmap_path = os.path.join(self.dataset_path, "Data", lightmap_names[t_key])
            lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
            lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3).astype(np.float32, copy=False))
        # 将各时间点拼接并保持为 float32
        lightmap_data_flat = np.concatenate(lightmap_in_different_time, axis=0).astype(np.float32, copy=False)

        # 读取 mask 序列，并堆叠成 (time_count, H, W)
        mask_in_different_time: List[np.ndarray] = []
        for t_key in times:
            mask_path = os.path.join(self.dataset_path, "Data", mask_names[t_key])
            # 使用无符号 uint8，避免 255 读成 -1 导致阈值判断失真
            mask_bin = np.fromfile(mask_path, dtype=np.uint8)
            mask_in_different_time.append(mask_bin.reshape(H, W))
        mask_data = np.stack(mask_in_different_time, axis=0)

        return lightmap_data_flat, mask_data

    def load_segmented_residuals(
        self,
        lightmap: Dict,
        times: List[str],
        day_baseline_key: str = "1200",
        night_baseline_key: str = "0",
        night_time_keys: List[str] | None = None,
        time_count: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载分段残差数据

        计算相对于基准时间点的残差，并按昼夜分段。

        Args:
            lightmap: 光照贴图信息
            times: 时间点列表
            day_baseline_key: 白天基准时间
            night_baseline_key: 夜晚基准时间
            night_time_keys: 夜晚时间点列表
            time_count: 时间点数量

        Returns:
            残差数据、遮罩、白天基准、夜晚基准、每时间基准、分段标记
        """
        night_set = set(night_time_keys or [])

        lightmap_flat, mask_data = self.load_lightmap_and_mask(lightmap, times)

        H, W = lightmap["resolution"]["height"], lightmap["resolution"]["width"]
        per_time = lightmap_flat.reshape(time_count, H * W, 3)

        day_idx = times.index(day_baseline_key)
        night_idx = times.index(night_baseline_key)
        baseline_day_flat = per_time[day_idx]
        baseline_night_flat = per_time[night_idx]

        baseline_per_time = np.empty_like(per_time)
        segment_flags = np.zeros(time_count, dtype=np.float32)

        for idx, time_key in enumerate(times):
            is_night = time_key in night_set
            segment_flags[idx] = 1.0 if is_night else 0.0
            baseline_per_time[idx] = baseline_night_flat if is_night else baseline_day_flat

        residual_per_time = per_time - baseline_per_time
        residual_flat = residual_per_time.reshape(time_count * H * W, 3).astype(np.float32, copy=False)
        np.nan_to_num(residual_flat, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)

        return (
            residual_flat,
            mask_data,
            baseline_day_flat.astype(np.float32, copy=False),
            baseline_night_flat.astype(np.float32, copy=False),
            baseline_per_time.astype(np.float32, copy=False),
            segment_flags,
        )

    def prepare_coords(
        self,
        resolution: Dict[str, int],
        times: List[str],
        segment_flags_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备训练坐标和分段标记

        Args:
            resolution: 分辨率信息
            times: 时间点列表
            segment_flags_tensor: 分段标记张量

        Returns:
            坐标张量和分段标记张量
        """
        H, W = resolution['height'], resolution['width']
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        coords_np = np.stack([ys / (H - 1), xs / (W - 1)], axis=-1).reshape(-1, 2)
        base_coords = torch.tensor(coords_np, dtype=torch.float32)
        coords_all: List[torch.Tensor] = []
        segment_all: List[torch.Tensor] = []
        for time_idx, t_key in enumerate(times):
            alpha = torch.full((H * W, 1), self.parse_times(t_key) / 24, dtype=torch.float32)
            coords_with_time = torch.cat([base_coords, alpha], dim=-1)
            coords_all.append(coords_with_time)

            segment_value = float(segment_flags_tensor[time_idx].item())
            segment_all.append(torch.full((H * W,), segment_value, dtype=torch.float32))

        coords_concat = torch.cat(coords_all, dim=0)
        segment_concat = torch.cat(segment_all, dim=0)
        return coords_concat, segment_concat

    @staticmethod
    def parse_times(time_str: str) -> float:
        """
        解析时间字符串为小时数

        Args:
            time_str: 时间字符串（如"1200"）

        Returns:
            小时数
        """
        return int(time_str) / 100.0

    def prepare_training_tensors(
        self,
        lightmap: Dict,
        times: List[str],
        day_baseline_key: str = "1200",
        night_baseline_key: str = "0",
        night_time_keys: List[str] | None = None,
        time_count: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, Dict[str, int]]:
        """
        准备训练张量

        封装完整的数据加载和预处理流程。

        Args:
            lightmap: 光照贴图信息
            times: 时间点列表
            day_baseline_key: 白天基准时间
            night_baseline_key: 夜晚基准时间
            night_time_keys: 夜晚时间点列表
            time_count: 时间点数量

        Returns:
            训练所需的各种张量和数据
        """
        residual_np_flat, mask_data, baseline_day_flat, baseline_night_flat, baseline_per_time_flat, segment_flags = self.load_segmented_residuals(
            lightmap, times, day_baseline_key, night_baseline_key, night_time_keys, time_count
        )

        residual_data = torch.tensor(residual_np_flat, dtype=torch.float32)
        baseline_per_time_tensor = torch.tensor(baseline_per_time_flat, dtype=torch.float32)
        segment_flags_tensor = torch.tensor(segment_flags, dtype=torch.float32)

        resolution = lightmap['resolution']
        coords_all, segment_per_sample = self.prepare_coords(resolution, times, segment_flags_tensor)

        mask_flat = torch.from_numpy(mask_data.reshape(-1))
        valid_mask = (mask_flat >= 127)

        coords_valid = coords_all[valid_mask]
        segments_valid = segment_per_sample[valid_mask]
        residual_valid = residual_data[valid_mask]

        # Clean up NaN/Inf values
        coords_valid = torch.nan_to_num(coords_valid, nan=0.0, posinf=1.0, neginf=0.0)
        residual_valid = torch.nan_to_num(residual_valid, nan=0.0, posinf=10.0, neginf=-10.0)

        return coords_valid, segments_valid, residual_valid, torch.tensor(baseline_day_flat, dtype=torch.float32), torch.tensor(baseline_night_flat, dtype=torch.float32), baseline_per_time_tensor, mask_data, resolution

    @staticmethod
    def default_times() -> List[str]:
        """
        获取默认时间点列表

        Returns:
            默认时间点字符串列表
        """
        return [
            "0", "100", "200", "300", "400", "500", "590", "600", "700", "800", "900", "1000",
            "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1810", "1900",
            "2000", "2100", "2200", "2300",
        ]

    @staticmethod
    def get_night_times() -> List[str]:
        """
        获取默认夜晚时间点列表

        Returns:
            夜晚时间点字符串列表
        """
        return [
            "0", "100", "200", "300", "400", "500", "590", "600", "1800", "1810", "1900",
            "2000", "2100", "2200", "2300",
        ]


