from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.encoding import encode_time, encode_xy


def _parse_levels(levels: Sequence[int] | str) -> List[int]:
    if isinstance(levels, str):
        parts = [p.strip() for p in levels.split(',') if p.strip()]
        return [int(p) for p in parts]
    else:
        return [int(v) for v in levels]


class MultiResGrid2D(nn.Module):
    """可学习的多分辨率2D特征网格集合。

    - levels: 各层分辨率（正方形网格，H=W=level）。
    - C: 每层通道数。
    - 初始化为小值的可训练参数，训练中通过双线性插值按像素采样。
    """
    def __init__(self, levels: Sequence[int] | str, channels_per_level: int = 16):
        super().__init__()
        self.levels = _parse_levels(levels)
        self.C = int(channels_per_level)
        params = []
        for res in self.levels:
            grid = nn.Parameter(torch.empty(self.C, res, res))
            # 小范围均匀初始化
            nn.init.uniform_(grid, -1e-3, 1e-3)
            params.append(grid)
        self.grids = nn.ParameterList(params)

    @property
    def out_dim(self) -> int:
        return len(self.levels) * self.C

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """按像素采样多级特征并拼接。
        xy: [N,2] in [-1,1] (x,y)
        return: [N, L*C]
        """
        assert xy.dim() == 2 and xy.size(-1) == 2, 'xy must be [N,2]'
        device = xy.device
        N = xy.shape[0]
        # grid_sample 期望 [B, H_out, W_out, 2]，我们构造 [1, N, 1, 2]
        grid = xy.view(1, N, 1, 2)
        outs = []
        for g in self.grids:
            feat = g.unsqueeze(0)  # [1,C,H,W]
            # align_corners=True 对应我们使用的 [-1,1] 归一化
            sampled = F.grid_sample(
                feat, grid, mode='bilinear', align_corners=True, padding_mode='border'
            )  # [1,C,N,1]
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # -> [N,C]
            outs.append(sampled)
        return torch.cat(outs, dim=-1)  # [N, L*C]


@dataclass
class GridMLPConfig:
    grid_levels: str = '16,32,64,128'
    channels_per_level: int = 16
    time_harmonics: int = 8
    # xy positional encoding controls
    xy_harmonics: int = 2            # 0 to disable Fourier encoding
    include_xy_input: bool = True    # True to also include raw xy
    mlp_hidden: int = 64
    mlp_layers: int = 3
    residual_mode: bool = True


class GridMLP(nn.Module):
    """多分辨率特征网格 + 小型 MLP，用时间做独立编码（方法B）。

    输入：
      - xy: [N,2] 归一化坐标（[-1,1]）
      - t_arr 或 t_feat: 若传 t_arr（标量/向量），内部用 encode_time 生成时间特征；或直接传 t_feat
    输出：
      - ΔRGB: [N,3]
    """
    def __init__(self, cfg: GridMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = MultiResGrid2D(cfg.grid_levels, cfg.channels_per_level)
        t_dim = 2 * max(0, int(cfg.time_harmonics))
        # xy feature dim: (2 if include_xy_input else 0) + 4*K
        xy_dim = (2 if bool(cfg.include_xy_input) else 0) + 4 * max(0, int(cfg.xy_harmonics))
        in_dim = self.grid.out_dim + xy_dim + t_dim

        layers: List[nn.Module] = []
        hidden = cfg.mlp_hidden
        depth = max(1, int(cfg.mlp_layers))
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        grid_feat = self.grid(xy)  # [N, L*C]
        # xy positional features (optional Fourier encoding and/or raw xy)
        if (self.cfg.xy_harmonics > 0) or self.cfg.include_xy_input:
            xy_feat = encode_xy(xy, K=int(self.cfg.xy_harmonics), include_input=bool(self.cfg.include_xy_input))
        else:
            xy_feat = xy.new_zeros((xy.shape[0], 0))
        feat = torch.cat([grid_feat, xy_feat, t_feat], dim=-1)
        return self.mlp(feat)

    @staticmethod
    def render_image(model: 'GridMLP', H: int, W: int, t_norm: float, device: torch.device, chunk: int = 1 << 18) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing='ij'
        )
        x_norm = (xs + 0.5) / W * 2 - 1
        y_norm = (ys + 0.5) / H * 2 - 1
        xy = torch.stack([x_norm, y_norm], dim=-1).view(-1, 2)
        N = xy.shape[0]
        # precompute temporal features once and expand to N
        t_vec = torch.full((N,), float(t_norm), device=device)
        t_feat = encode_time(t_vec, K=model.cfg.time_harmonics)
        outs = []
        with torch.no_grad():
            for i in range(0, N, chunk):
                j = min(i + chunk, N)
                pred = model(xy[i:j], t_feat=t_feat[i:j])
                outs.append(pred)
        out = torch.cat(outs, dim=0).view(H, W, 3)
        return out
