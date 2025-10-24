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
