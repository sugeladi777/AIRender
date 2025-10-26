from __future__ import annotations
"""
轻量级特征编码器（预留）：
- BaselineEncoder: 将基准图像（H,W,3）编码为浅层特征图（C,H',W'），便于后续按像素采样并交给小 MLP 预测 ΔRGB。
当前未在训练脚本中默认启用，作为后续改进的模块边界预留。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class BaselineEncoder(nn.Module):
    """将 RGB 基准图编码为浅层特征图。

    参数：
      - in_ch: 输入通道（RGB=3）
      - base_ch: 基础通道数（默认 16）
      - out_ch: 输出通道数（默认 32）
      - downsample: 下采样倍率（1/2/4），>1 时通过 stride 实现
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 16, out_ch: int = 32, downsample: int = 1):
        super().__init__()
        s1 = 2 if downsample >= 2 else 1
        s2 = 2 if downsample >= 4 else 1
        self.stage1 = ConvBlock(in_ch, base_ch, k=3, s=s1, p=1)
        self.stage2 = ConvBlock(base_ch, base_ch, k=3, s=1, p=1)
        self.stage3 = ConvBlock(base_ch, out_ch, k=3, s=s2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W], 输出 [B,C,H',W']
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
