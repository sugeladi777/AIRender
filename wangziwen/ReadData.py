import json
import os
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------
# Data loading helpers
# -----------------------------

def load_dataset_config(dataset_path: str, config_file: str = "config.json") -> Dict:
    # 读取数据集配置文件并返回字典。
    config_path = os.path.join(dataset_path, config_file)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expected_lightmap_bytes(height: int, width: int) -> int:
    # 每个像素 3 通道 float32（4 字节）
    return height * width * 3 * 4


def load_lightmap_and_mask(
    dataset_path: str,
    lightmap: Dict,
    times: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    # 读取单个 lightmap 在多个时间点的光照图与 mask。

    lightmap_names = lightmap["lightmaps"]
    mask_names = lightmap["masks"]
    resolution = lightmap["resolution"]

    H, W = resolution["height"], resolution["width"]
    expected_bytes = _expected_lightmap_bytes(H, W)

    # 读取 lightmap 序列并展平为 (H*W,3)，随后在时间维拼接
    lightmap_in_different_time: List[np.ndarray] = []
    for t_key in times:
        lightmap_path = os.path.join(dataset_path, "Data", lightmap_names[t_key])
        lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
        lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3))
    lightmap_data_flat = np.concatenate(lightmap_in_different_time, axis=0).astype(np.float32, copy=False)

    # 读取 mask 序列，并堆叠成 (time_count, H, W)
    mask_in_different_time: List[np.ndarray] = []
    for t_key in times:
        mask_path = os.path.join(dataset_path, "Data", mask_names[t_key])
        # 使用无符号 uint8，避免 255 读成 -1 导致阈值判断失真
        mask_bin = np.fromfile(mask_path, dtype=np.uint8)
        mask_in_different_time.append(mask_bin.reshape(H, W))
    mask_data = np.stack(mask_in_different_time, axis=0)

    return lightmap_data_flat, mask_data


def default_times() -> List[str]:
    # 提供默认时间列表，便于外部快速复用（可选）。
    return [
        "0",
        "100",
        "200",
        "300",
        "400",
        "500",
        "590",
        "600",
        "700",
        "800",
        "900",
        "1000",
        "1100",
        "1200",
        "1300",
        "1400",
        "1500",
        "1600",
        "1700",
        "1800",
        "1810",
        "1900",
        "2000",
        "2100",
        "2200",
        "2300",
    ]

def get_night_times() -> List[str]:
    # 提供默认夜晚时间列表，便于外部快速复用（可选）。
    return [
        "0",
        "100",
        "200",
        "300",
        "400",
        "500",
        "590",
        "600",
        "1800",
        "1810",
        "1900",
        "2000",
        "2100",
        "2200",
        "2300",
    ]

def load_segmented_residuals(
    dataset_path: str,
    lightmap: Dict,
    times: List[str],
    day_baseline_key: str = "1200",
    night_baseline_key: str = "0",
    night_time_keys: List[str] | None = None,
    time_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取残差并按昼夜分段，返回对应基准图与标记。"""
    night_set = set(night_time_keys or [])

    lightmap_flat, mask_data = load_lightmap_and_mask(dataset_path, lightmap, times)

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

    return (
        residual_flat,
        mask_data,
        baseline_day_flat.astype(np.float32, copy=False),
        baseline_night_flat.astype(np.float32, copy=False),
        baseline_per_time.astype(np.float32, copy=False),
        segment_flags,
    )


