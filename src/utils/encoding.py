from __future__ import annotations
import numpy as np
import torch


def encode_time(t_norm: float | np.ndarray | torch.Tensor, K: int = 2):
    """
    时间 Fourier 编码：把归一化时间 t_norm (在 [0,1]) 映射为 sin/cos 特征。

    输出维度为 2*K（对每个 k 返回 sin 和 cos）。函数支持 numpy 数组或 torch 张量输入。
    """
    # 依据输入类型选择常量
    two_pi = 2.0 * np.pi if not isinstance(t_norm, torch.Tensor) else 2.0 * torch.pi

    def _encode_np(tn: np.ndarray) -> np.ndarray:
        outs = []
        for k in range(1, K + 1):
            outs.append(np.sin(two_pi * k * tn))
            outs.append(np.cos(two_pi * k * tn))
        return np.stack(outs, axis=-1).astype(np.float32)

    def _encode_torch(tn: torch.Tensor) -> torch.Tensor:
        outs = []
        for k in range(1, K + 1):
            outs.append(torch.sin(two_pi * k * tn))
            outs.append(torch.cos(two_pi * k * tn))
        return torch.stack(outs, dim=-1).to(dtype=torch.float32)

    if isinstance(t_norm, torch.Tensor):
        return _encode_torch(t_norm)
    else:
        tn = np.array(t_norm, dtype=np.float32)
        return _encode_np(tn)


def encode_xy(xy: np.ndarray | torch.Tensor, K: int = 0, include_input: bool = True):
    """
    2D 坐标 (x,y) 的 Fourier 位置编码。

    参数：
    - xy: [..., 2]，范围通常在 [-1,1]
    - K: 频率数；当 K=0 时返回原始 xy（若 include_input=True）或空特征（False）
    - include_input: 是否把原始 xy 拼接到输出前面

    返回：与 xy 同批量形状的特征，最后一维为 (2 if include_input else 0) + 4*K
    编码形式：对每个 k=1..K，拼接 [sin(2πk x), cos(2πk x), sin(2πk y), cos(2πk y)]
    """
    if K <= 0:
        if include_input:
            return xy if isinstance(xy, torch.Tensor) else np.array(xy, dtype=np.float32)
        # 返回空特征
        if isinstance(xy, torch.Tensor):
            return xy.new_zeros((*xy.shape[:-1], 0))
        else:
            shp = np.array(xy).shape
            return np.zeros((*shp[:-1], 0), dtype=np.float32)

    if isinstance(xy, torch.Tensor):
        xy = xy.to(dtype=torch.float32)
        x = xy[..., 0]
        y = xy[..., 1]
        two_pi = 2.0 * torch.pi
        blocks = []
        if include_input:
            blocks.append(xy)
        for k in range(1, K + 1):
            blk = torch.stack([
                torch.sin(two_pi * k * x),
                torch.cos(two_pi * k * x),
                torch.sin(two_pi * k * y),
                torch.cos(two_pi * k * y),
            ], dim=-1)  # [..., 4]
            blocks.append(blk)
        return torch.cat(blocks, dim=-1)  # [..., 2 + 4K]
    else:
        xy = np.array(xy, dtype=np.float32)
        x = xy[..., 0]
        y = xy[..., 1]
        two_pi = 2.0 * np.pi
        outs = []
        if include_input:
            outs.append(xy)
        for k in range(1, K + 1):
            blk = np.stack([
                np.sin(two_pi * k * x),
                np.cos(two_pi * k * x),
                np.sin(two_pi * k * y),
                np.cos(two_pi * k * y),
            ], axis=-1)  # [..., 4]
            outs.append(blk)
        return np.concatenate(outs, axis=-1).astype(np.float32)
