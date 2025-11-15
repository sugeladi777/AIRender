from __future__ import annotations

from typing import Dict
import numpy as np
import torch

from Model import Model
from ReadData import default_times, get_night_times


class BasicInterface:
    def __init__(self, lightmap_config: Dict, device: torch.device, hidden_dim: int = 256):
        self.device = device
        self.times = default_times()
        self.night_times = set(get_night_times())

        # 模型输入 (y_norm, x_norm, t_norm, is_night)
        self.model = Model(input_dim=4, output_dim=3, hidden_dim=hidden_dim).to(device)
        self.model.eval()

        # 解析分辨率与参数文件路径
        resolution = lightmap_config['resolution']
        self.height = resolution['height']
        self.width = resolution['width']
        level = lightmap_config.get('level', 'level0')
        lid = lightmap_config.get('id', '1')
        param_path = f"./Parameters/model_{level}_{lid}_params.bin"

        # 加载模型参数（前半段）+ 基准图（后半段，H*W*3）
        params_array = np.fromfile(param_path, dtype=np.float32)
        num_params = sum(p.numel() for p in self.model.parameters())
        with torch.no_grad():
            offset = 0
            for param in self.model.parameters():
                size = param.numel()
                param_data = params_array[offset:offset + size]
                param.copy_(torch.from_numpy(param_data.reshape(param.shape)))
                offset += size

        expected_baseline_len = self.height * self.width * 3
        remaining = params_array.size - num_params
        if remaining >= 2 * expected_baseline_len:
            baseline_day_flat = params_array[num_params:num_params + expected_baseline_len]
            baseline_night_flat = params_array[num_params + expected_baseline_len:num_params + 2 * expected_baseline_len]
        elif remaining >= expected_baseline_len:
            baseline_day_flat = params_array[num_params:num_params + expected_baseline_len]
            baseline_night_flat = baseline_day_flat
        else:
            baseline_day_flat = np.zeros(expected_baseline_len, dtype=np.float32)
            baseline_night_flat = baseline_day_flat

        self.baseline_day = torch.from_numpy(baseline_day_flat.reshape(self.height * self.width, 3)).float().to(self.device)
        self.baseline_night = torch.from_numpy(baseline_night_flat.reshape(self.height * self.width, 3)).float().to(self.device)

    def reconstruct(self, current_time: float) -> torch.Tensor:
        """重建给定时间点的整张图像，返回 [1,3,H,W]。"""
        H, W = self.height, self.width
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij',
        )
        alpha = float(current_time) / 24.0
        is_night = self._segment_indicator(current_time) > 0.5
        segment_flag = torch.full((H * W, 1), 1.0 if is_night else 0.0, device=self.device, dtype=torch.float32)
        coords = torch.stack([
            ys.ravel() / (H - 1),
            xs.ravel() / (W - 1),
            torch.full((H * W,), alpha, device=self.device)
        ], dim=-1).float()
        coords = torch.cat([coords, segment_flag], dim=-1)

        with torch.no_grad():
            residual = self.model(coords)
        baseline = self.baseline_night if is_night else self.baseline_day
        result = residual + baseline
        result = result.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        # cache last result so external code (tests) can fetch it
        self._last_result = result
        return result

    def get_result(self) -> torch.Tensor:
        """返回最近一次 reconstruct 的结果 [1,3,H,W]。
        在没有 reconstruct 调用时会返回 None。
        """
        return getattr(self, "_last_result", None)

    def random_test(self, coord: torch.Tensor) -> torch.Tensor:
        """查询单像素的重建结果。

        coord: [1,3] -> (y, x, time)，y/x 为像素坐标（非归一化），time ∈ [0,24]
        返回: [1,3]
        """
        coord = torch.as_tensor(coord, dtype=torch.float32, device=self.device)
        y = int(coord[0, 0].item())
        x = int(coord[0, 1].item())
        t = float(coord[0, 2].item())
        y = max(0, min(self.height - 1, y))
        x = max(0, min(self.width - 1, x))
        yn = y / (self.height - 1)
        xn = x / (self.width - 1)
        alpha = t / 24.0
        is_night = self._segment_indicator(t) > 0.5
        segment_flag = torch.tensor([[1.0 if is_night else 0.0]], dtype=torch.float32, device=self.device)
        input_coord = torch.tensor([[yn, xn, alpha]], dtype=torch.float32, device=self.device)
        input_coord = torch.cat([input_coord, segment_flag], dim=-1)
        with torch.no_grad():
            residual = self.model(input_coord)
        base = (self.baseline_night if is_night else self.baseline_day)[y * self.width + x].unsqueeze(0)
        return residual + base

    def _segment_indicator(self, current_time: float) -> float:
        closest_time = min(self.times, key=lambda key: abs(self._parse_time(key) - current_time)) if self.times else "1200"
        return 1.0 if closest_time in self.night_times else 0.0

    @staticmethod
    def _parse_time(key: str) -> float:
        return int(key) / 100.0


def get(lightmap_config: Dict, device: torch.device, hidden_dim: int = 256) -> BasicInterface:
    return BasicInterface(lightmap_config, device, hidden_dim)
