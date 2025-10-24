import math
from typing import List
import torch
import torch.nn as nn


class Sine(nn.Module):
    """简单的正弦激活封装（带频率因子 omega_0）。

    用于快速替换或测试，但在 SIREN 中常直接使用 SineLayer。
    """
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SineLayer(nn.Module):
    """
    SIREN 的一层实现：Linear -> sin(omega0 * x)

    包含 SIREN 推荐的初始化方法：
    - 第一层使用较小的均匀分布 U(-1/in, 1/in)
    - 后续层使用 U(-sqrt(6/in)/omega0, sqrt(6/in)/omega0)
    这样可以在训练初期避免输出变得过大或过小。
    """
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        # 使用无梯度初始化权重
        with torch.no_grad():
            if self.is_first:
                # 首层较窄的初始化范围
                bound = 1 / self.in_features
            else:
                # 隐藏层按 SIREN 论文的建议缩放
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 线性层后乘 omega0 再做 sin
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """多层 SIREN MLP 封装：由若干 SineLayer + 一个线性输出组成。

    参数：
    - in_features: 输入维度
    - hidden_features: 隐藏层宽度
    - hidden_layers: 隐藏层数量（含首层）
    - out_features: 输出维度
    - outermost_activation: 最外层可选激活（例如 Sigmoid 用于 [0,1] 输出）
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_activation: nn.Module | None = None,
    ):
        super().__init__()
        layers: List[nn.Module] = []

        # 第一层使用 is_first=True 的 SineLayer
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        # 隐藏层（可能为0时只会有首层 + 最后一线性输出）
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        # 最后一层为线性层（不加 sin），通常后面可接 outermost_activation
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            # 线性层按隐藏层相同尺度初始化以保持稳定
            bound = math.sqrt(6 / hidden_features) / hidden_omega_0
            nn.init.uniform_(final_linear.weight, -bound, bound)
            nn.init.uniform_(final_linear.bias, -bound, bound)
        layers.append(final_linear)

        if outermost_activation is not None:
            layers.append(outermost_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向直接串联执行
        return self.net(x)
