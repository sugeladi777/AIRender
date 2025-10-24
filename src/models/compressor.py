from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .siren import SirenMLP
from ..utils.encoding import encode_xy


class F1Net(nn.Module):
    """
    F1 网络：将归一化坐标 (x,y) 映射为像素的潜变量 z。

    说明：
    - 输入：二维坐标，范围为 [-1,1]
    - 输出：latent_dim 维的向量，可视为每像素的压缩表示
    """
    def __init__(self, latent_dim: int = 32, hidden: int = 32, layers: int = 4,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0,
                 xy_harmonics: int = 0, xy_include_input: bool = True):
        super().__init__()
        self.xy_harmonics = xy_harmonics
        self.xy_include_input = xy_include_input
        in_features = (2 if xy_include_input else 0) + (4 * xy_harmonics)
        if in_features == 0:
            # 退化保护：至少保留原始 xy
            in_features = 2
            self.xy_include_input = True
        # 使用 SIREN MLP（sin 激活）以便拟合高频细节
        self.mlp = SirenMLP(
            in_features=in_features,
            hidden_features=hidden,
            hidden_layers=layers,
            out_features=latent_dim,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            outermost_activation=None,
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: [N, 2] in [-1, 1]
        if self.xy_harmonics > 0 or not self.xy_include_input:
            feat = encode_xy(xy, K=self.xy_harmonics, include_input=self.xy_include_input)
        else:
            feat = xy
        return self.mlp(feat)


class F2Net(nn.Module):
    """
    F2 网络：将潜变量 z 与时间特征拼接后映射为 RGB 颜色。

    说明：
    - 输入：z + time_feat
    - 输出：3 维 RGB 在 [0,1]（使用 Sigmoid）
    """
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 4,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()
        # 最外层使用 Sigmoid 将输出限制到 [0,1]
        self.mlp = SirenMLP(
            in_features=in_dim,
            hidden_features=hidden,
            hidden_layers=layers,
            out_features=3,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            outermost_activation=nn.Sigmoid(),  # Clamp to [0,1]
        )

    def forward(self, zt: torch.Tensor) -> torch.Tensor:
        # zt: [N, in_dim]
        return self.mlp(zt)


class F1F2Model(nn.Module):
    """
    组合模型：F1 + F2，用于端到端训练。

    forward(xy, t_feat) -> rgb, latent
    """
    def __init__(self,
                 latent_dim: int = 32,
                 f1_hidden: int = 32,
                 f1_layers: int = 4,
                 f2_hidden: int = 64,
                 f2_layers: int = 4,
                 time_feat_dim: int = 4,
                 omega0_first: float = 30.0,
                 omega0_hidden: float = 30.0,
                 xy_harmonics: int = 0,
                 xy_include_input: bool = True):
        super().__init__()
        # F1: 坐标 -> latent
        self.f1 = F1Net(latent_dim, f1_hidden, f1_layers, omega0_first, omega0_hidden,
                        xy_harmonics=xy_harmonics, xy_include_input=xy_include_input)
        # F2: (latent + time) -> rgb
        self.f2 = F2Net(latent_dim + time_feat_dim, f2_hidden, f2_layers, omega0_first, omega0_hidden)
        self.latent_dim = latent_dim
        self.time_feat_dim = time_feat_dim
        self.xy_harmonics = xy_harmonics
        self.xy_include_input = xy_include_input

    def forward(self, xy: torch.Tensor, t_feat: torch.Tensor):
        # xy: [N, 2], t_feat: [N, time_feat_dim]
        z = self.f1(xy)
        rgb = self.f2(torch.cat([z, t_feat], dim=-1))
        return rgb, z

    @torch.no_grad()
    def compute_latent_map(self, H: int, W: int, device: torch.device, batch: int = 1 << 20) -> torch.Tensor:
        """计算整张图的 latent_map（分批推理以节省内存）。

        返回形状为 [H, W, latent_dim] 的 Tensor（CPU 上通常会返回 float32）。
        """
        self.eval()
        # 生成像素坐标网格，归一化到 [-1,1]
        xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2 - 1
        ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2 - 1
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [H*W, 2]

        latents = []
        # 分批处理，避免一次性占满显存/内存
        for i in range(0, coords.shape[0], batch):
            chunk = coords[i:i+batch]
            z = self.f1(chunk)
            latents.append(z.cpu())
        latent = torch.cat(latents, dim=0).reshape(H, W, self.latent_dim)
        return latent
