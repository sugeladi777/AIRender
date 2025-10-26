from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .siren import SirenMLP
from ..utils.encoding import encode_xy, encode_time


@dataclass
class DeltaFieldConfig:
    time_harmonics: int = 4
    xy_harmonics: int = 4
    xy_include_input: bool = True
    hidden: int = 64
    layers: int = 6
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0


class DeltaField(nn.Module):
    """
    一个直接学习 f(x,y,t)->ΔRGB 的场（残差，范围可在 [-1,1]）。

    输入:
      - xy: [N,2] 归一化到 [-1,1]
      - t_feat: [N, 2*K_t] 由 encode_time 得到
    输出:
      - delta_rgb: [N,3]
    """
    def __init__(self, cfg: DeltaFieldConfig):
        super().__init__()
        self.cfg = cfg
        xy_dim = (2 if cfg.xy_include_input else 0) + 4 * max(0, int(cfg.xy_harmonics))
        t_dim = 2 * max(0, int(cfg.time_harmonics))
        in_dim = xy_dim + t_dim
        self.mlp = SirenMLP(
            in_features=in_dim,
            hidden_features=cfg.hidden,
            hidden_layers=cfg.layers,
            out_features=3,
            first_omega_0=cfg.first_omega_0,
            hidden_omega_0=cfg.hidden_omega_0,
            outermost_activation=None,
        )

    def forward(self, xy: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        # xy: [N,2], t_feat: [N,2*K]
        xye = encode_xy(xy, K=self.cfg.xy_harmonics, include_input=self.cfg.xy_include_input)
        if not isinstance(xye, torch.Tensor):
            xye = torch.from_numpy(xye).to(xy.device)
        feat = torch.cat([xye, t_feat], dim=-1)
        return self.mlp(feat)

    @staticmethod
    def render_image(model: 'DeltaField', H: int, W: int, t_norm: float, device: torch.device, chunk: int = 1<<18) -> torch.Tensor:
        """渲染整张图像的 ΔRGB。
        返回: [H,W,3] torch.float32，未裁剪。
        """
        ys, xs = torch.meshgrid(
            torch.linspace(0, H-1, H, device=device),
            torch.linspace(0, W-1, W, device=device),
            indexing='ij')
        x_norm = (xs + 0.5) / W * 2 - 1
        y_norm = (ys + 0.5) / H * 2 - 1
        xy = torch.stack([x_norm, y_norm], dim=-1).view(-1, 2)
        N = xy.shape[0]
        # time features
        t_arr = torch.full((N,), float(t_norm), device=device)
        t_feat = encode_time(t_arr, K=model.cfg.time_harmonics)
        if not isinstance(t_feat, torch.Tensor):
            t_feat = torch.from_numpy(t_feat).to(device)
        outs = []
        with torch.no_grad():
            for i in range(0, N, chunk):
                j = min(i + chunk, N)
                pred = model(xy[i:j], t_feat[i:j])  # [M,3]
                outs.append(pred)
        out = torch.cat(outs, dim=0).view(H, W, 3)
        return out
