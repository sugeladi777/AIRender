"""
Interface.py: 光照贴图重建接口

提供统一的接口用于加载训练好的模型并重建任意时间点的光照贴图。
支持白天/夜晚模型切换和单像素查询。
"""

from __future__ import annotations

from typing import Dict
import os
import numpy as np
import torch

from Model import Model
from Dataset import LightmapDataLoader


class BasicInterface:
    """
    光照贴图重建接口

    加载白天和夜晚模型，预计算编码，提供重建和查询功能。
    """

    def __init__(self, lightmap_config: Dict, device: torch.device, hidden_dim: int = 256):
        """
        初始化接口

        Args:
            lightmap_config: 光照贴图配置字典
            device: 计算设备
            hidden_dim: 模型隐藏层维度
        """
        self.device = device
        self.times = LightmapDataLoader.default_times()
        self.night_times = set(LightmapDataLoader.get_night_times())

        # 两个模型分别拟合白天/夜晚残差，输入为 (y_norm, x_norm, t_norm)
        self.day_model = Model(output_dim=3, hidden_dim=hidden_dim).to(device)
        self.night_model = Model(output_dim=3, hidden_dim=hidden_dim).to(device)
        self.day_model.eval()
        self.night_model.eval()

        # 解析分辨率与参数文件路径
        resolution = lightmap_config['resolution']
        self.height = resolution['height']
        self.width = resolution['width']
        level = lightmap_config.get('level', 'level0')
        lid = lightmap_config.get('id', '1')
        expected_baseline_len = self.height * self.width * 3

        day_path = f"./Parameters/model_{level}_{lid}_day_params.bin"
        night_path = f"./Parameters/model_{level}_{lid}_night_params.bin"

        self.baseline_day = self._load_model_and_baseline(self.day_model, day_path, expected_baseline_len)
        self.baseline_night = self._load_model_and_baseline(self.night_model, night_path, expected_baseline_len)
        # 预计算 spatial 编码（每个像素一次）以及每个时间点的 time 编码向量（很小）
        # 我们只缓存 spatial encoder 的输出和 time encoder 的向量，推理时再把两者 concat 并送入 MLP
        self._precompute_encodings()

    def _load_model_and_baseline(
        self,
        model: Model,
        param_path: str,
        baseline_len: int,
    ) -> torch.Tensor:
        """
        加载模型参数和基准数据

        Args:
            model: 要加载参数的模型
            param_path: 参数文件路径
            baseline_len: 基准数据长度

        Returns:
            基准数据张量
        """
        params_array = np.fromfile(param_path, dtype=np.float32) if os.path.exists(param_path) else np.array([], dtype=np.float32)
        num_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            offset = 0
            for param in model.parameters():
                size = param.numel()
                if offset + size <= params_array.size:
                    param_data = params_array[offset:offset + size]
                    param.copy_(torch.from_numpy(param_data.reshape(param.shape)))
                else:
                    param.zero_()
                offset += size

        baseline = np.zeros(baseline_len, dtype=np.float32)
        if params_array.size >= num_params + baseline_len:
            baseline = params_array[num_params:num_params + baseline_len]
        elif params_array.size > num_params:
            available = params_array.size - num_params
            baseline[:available] = params_array[num_params:params_array.size]

        return torch.from_numpy(baseline.reshape(self.height * self.width, 3)).float().to(self.device)

    def reconstruct(self, current_time: float) -> torch.Tensor:
        """
        重建给定时间点的整张图像

        Args:
            current_time: 时间点（小时，0-24）

        Returns:
            重建的图像张量 [1, 3, H, W]
        """
        # 使用缓存的 spatial encoding 与 time vector 构建 encoded features，直接走 MLP
        H, W = self.height, self.width
        is_night = self._segment_indicator(current_time) > 0.5
        closest_time = min(self.times, key=lambda key: abs(self._parse_time(key) - current_time)) if self.times else "1200"
        time_enc = self._time_enc_map[closest_time]

        # 将 time_enc 扩展到每个像素并与 spatial_enc concat
        with torch.no_grad():
            time_expanded = time_enc.unsqueeze(0).expand(self._spatial_enc.shape[0], -1)
            encoded = torch.cat([self._spatial_enc, time_expanded], dim=-1)
            model = self.night_model if is_night else self.day_model
            residual = model(encoded)

        baseline = self.baseline_night if is_night else self.baseline_day
        result = residual + baseline
        result = result.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0)
        # cache last result so external code (tests) can fetch it
        self._last_result = result.cpu()  # Move to CPU for compatibility with numpy indexing
        return result

    def get_result(self) -> torch.Tensor:
        """
        获取最近一次重建的结果

        Returns:
            最近的重建结果 [1, 3, H, W]，如果没有则返回None
        """
        return getattr(self, "_last_result", None)

    def random_test(self, coord: torch.Tensor) -> torch.Tensor:
        """
        查询单像素的重建结果

        Args:
            coord: 坐标 [1, 3] -> (y, x, time)，y/x为像素坐标，time∈[0,24]

        Returns:
            单像素结果 [1, 3]
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
        closest_time = min(self.times, key=lambda key: abs(self._parse_time(key) - t)) if self.times else "1200"

        # 直接使用 encoder 来计算单像素的 encoded 表示，然后调用 MLP
        with torch.no_grad():
            spatial_coord = torch.tensor([[yn, xn]], dtype=torch.float32, device=self.device)
            spatial_enc = self.day_model.spatial_encoder(spatial_coord)
            time_enc = self._time_enc_map[closest_time].unsqueeze(0)
            encoded = torch.cat([spatial_enc, time_enc], dim=-1)
            model = self.night_model if is_night else self.day_model
            residual = model(encoded)

        base = (self.baseline_night if is_night else self.baseline_day)[y * self.width + x].unsqueeze(0)
        return residual + base

    def _precompute_encodings(self) -> None:
        """
        预计算所有像素的空间编码和每个离散时间点的时间编码向量

        缓存空间编码 (H*W, Sdim) 和时间编码映射 {time_key: (Tdim,)}
        重建时扩展时间向量并与空间编码拼接后送入模型
        """
        H, W = self.height, self.width
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij',
        )
        spatial_coords = torch.stack([ys.ravel() / (H - 1), xs.ravel() / (W - 1)], dim=-1).float().to(self.device)

        with torch.no_grad():
            # 使用 day_model 的 encoder（两个模型结构相同），只计算一次 spatial encoding
            self.day_model.eval()
            self._spatial_enc = self.day_model.spatial_encoder(spatial_coords)
            self._time_enc_map = {}
            for t_key in self.times:
                alpha = float(self._parse_time(t_key)) / 24.0
                t_in = torch.tensor([[alpha]], dtype=torch.float32, device=self.device)
                t_enc = self.day_model.time_encoder(t_in).squeeze(0)
                self._time_enc_map[t_key] = t_enc

    def _segment_indicator(self, current_time: float) -> float:
        """
        获取时间点的昼夜指示器

        Args:
            current_time: 时间点（小时）

        Returns:
            1.0表示夜晚，0.0表示白天
        """
        closest_time = min(self.times, key=lambda key: abs(self._parse_time(key) - current_time)) if self.times else "1200"
        return 1.0 if closest_time in self.night_times else 0.0

    @staticmethod
    def _parse_time(key: str) -> float:
        """
        解析时间字符串为小时数

        Args:
            key: 时间字符串（如"1200"）

        Returns:
            小时数
        """
        return int(key) / 100.0


def get(lightmap_config: Dict, device: torch.device, hidden_dim: int = 256) -> BasicInterface:
    """
    创建接口实例的工厂函数

    Args:
        lightmap_config: 光照贴图配置
        device: 计算设备
        hidden_dim: 隐藏层维度

    Returns:
        接口实例
    """
    return BasicInterface(lightmap_config, device, hidden_dim)
