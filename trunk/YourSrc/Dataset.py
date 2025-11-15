import json
import os
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch


class LightmapDataset:
    """封装lightmap读取，提供训练所需的tensor"""

    def __init__(self, root: str, config_name: str = "config.json", time_count: Optional[int] = None) -> None:
        self.root = root
        self.config_path = os.path.join(root, config_name)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.time_count = time_count
        self.lightmaps: List[Dict] = self.config.get("lightmap_list", [])
        if not self.lightmaps:
            raise ValueError("lightmap_list is empty in config")

    def __len__(self) -> int:
        return len(self.lightmaps)

    def __iter__(self) -> Iterator[Dict]:
        for lightmap_cfg in self.lightmaps:
            yield self._prepare_sample(lightmap_cfg)

    def _prepare_sample(self, lightmap_cfg: Dict) -> Dict:
        time_keys = self._resolve_time_keys(lightmap_cfg)
        resolution = lightmap_cfg["resolution"]
        height, width = resolution["height"], resolution["width"]
        lightmap_tensor = self._load_lightmap(lightmap_cfg, time_keys, height, width)
        mask_array = self._load_mask(lightmap_cfg, time_keys, height, width)
        coords_tensor = self._build_coords(len(time_keys), height, width)
        return {
            "id": lightmap_cfg["id"],
            "level": lightmap_cfg["level"],
            "resolution": resolution,
            "time_count": len(time_keys),
            "coords": coords_tensor,
            "lightmap": lightmap_tensor,
            "mask": mask_array,
        }

    def _resolve_time_keys(self, lightmap_cfg: Dict) -> List[str]:
        keys = sorted(lightmap_cfg["lightmaps"].keys(), key=lambda k: int(k))
        if self.time_count is not None:
            if self.time_count <= 0:
                raise ValueError("time_count must be positive")
            keys = keys[: self.time_count]
        return keys

    def _load_lightmap(
        self,
        lightmap_cfg: Dict,
        time_keys: List[str],
        height: int,
        width: int,
    ) -> torch.Tensor:
        # 逐时间帧读取lightmap
        frames = []
        for key in time_keys:
            path = os.path.join(self.root, lightmap_cfg["lightmaps"][key])
            data = np.fromfile(path, dtype=np.float32)
            frames.append(data.reshape(height * width, 3))
        stacked = np.concatenate(frames, axis=0)
        return torch.from_numpy(stacked).to(torch.float32)

    def _load_mask(
        self,
        lightmap_cfg: Dict,
        time_keys: List[str],
        height: int,
        width: int,
    ) -> np.ndarray:
        # 逐时间帧读取mask
        masks = []
        for key in time_keys:
            path = os.path.join(self.root, lightmap_cfg["masks"][key])
            data = np.fromfile(path, dtype=np.int8)
            masks.append(data.reshape(height, width))
        return np.stack(masks, axis=0)

    def _build_coords(self, time_count: int, height: int, width: int) -> torch.Tensor:
        # 构造(y, x, t)归一化坐标
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        base = np.stack([grid_y, grid_x], axis=-1).reshape(-1, 2)
        coords = []
        denom = max(1, time_count - 1)
        for idx in range(time_count):
            alpha = np.full((height * width, 1), idx / denom, dtype=np.float32)
            coords.append(np.concatenate([base, alpha], axis=-1))
        return torch.from_numpy(np.concatenate(coords, axis=0)).to(torch.float32)
