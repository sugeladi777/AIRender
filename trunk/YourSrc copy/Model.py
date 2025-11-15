"""
Model.py: 光照贴图残差重建神经网络模型

基于PyTorch实现的MLP模型，使用傅里叶特征编码处理空间和时间坐标。
支持白天/夜晚模型的残差预测，用于光照贴图重建任务。
"""

from typing import Literal, List

import math
import torch
import torch.nn as nn


class _FourierFeatureEncoding(nn.Module):
    """
    傅里叶特征编码模块

    将输入坐标编码为正弦和余弦特征，用于提高神经网络对高频信号的建模能力。
    """

    def __init__(
        self,
        in_dims: int,
        n_frequencies: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
        """
        初始化傅里叶编码器

        Args:
            in_dims: 输入维度
            n_frequencies: 频率数量
            include_input: 是否包含原始输入
            log_sampling: 是否使用对数采样
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_frequencies = max(0, n_frequencies)
        self.include_input = include_input
        if self.n_frequencies > 0:
            if log_sampling:
                freq_bands = 2.0 ** torch.arange(self.n_frequencies, dtype=torch.float32)
            else:
                freq_bands = torch.linspace(1.0, float(self.n_frequencies), self.n_frequencies)
            self.register_buffer("freq_bands", freq_bands)
        else:
            self.register_buffer("freq_bands", torch.empty(0))

    @property
    def out_dim(self) -> int:
        """输出维度"""
        base = self.in_dims if self.include_input else 0
        return base + 2 * self.in_dims * self.n_frequencies

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: 输入张量

        Returns:
            编码后的特征
        """
        out_parts = []
        if self.include_input:
            out_parts.append(inputs)
        if self.n_frequencies > 0:
            # (..., dims, n_freq)
            expanded = inputs.unsqueeze(-1) * (2.0 * math.pi * self.freq_bands)
            sin_enc = torch.sin(expanded)
            cos_enc = torch.cos(expanded)
            out_parts.append(sin_enc.reshape(*inputs.shape[:-1], -1))
            out_parts.append(cos_enc.reshape(*inputs.shape[:-1], -1))
        return torch.cat(out_parts, dim=-1) if out_parts else torch.zeros_like(inputs)

class Model(nn.Module):
    """
    光照贴图残差重建模型

    使用傅里叶特征编码的MLP网络，输入归一化的(y,x,t)坐标，
    输出RGB残差值用于重建光照贴图。
    """

    def __init__(
        self,
        output_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 5,
        spatial_frequencies: int = 8,
        time_frequencies: int = 4,
        include_input: bool = True,
        log_sampling: bool = True,
        activation: Literal["ReLU"] = "ReLU",
        output_activation: Literal["None", "Sigmoid"] = "None",
    ) -> None:
        """
        初始化模型

        Args:
            output_dim: 输出维度（RGB为3）
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层层数
            spatial_frequencies: 空间频率编码数量
            time_frequencies: 时间频率编码数量
            include_input: 是否在编码中包含原始输入
            log_sampling: 是否使用对数频率采样
            activation: 隐藏层激活函数
            output_activation: 输出激活函数
        """
        super().__init__()

        self.spatial_encoder = _FourierFeatureEncoding(
            in_dims=2,
            n_frequencies=spatial_frequencies,
            include_input=include_input,
            log_sampling=log_sampling,
        )
        self.time_encoder = _FourierFeatureEncoding(
            in_dims=1,
            n_frequencies=time_frequencies,
            include_input=include_input,
            log_sampling=log_sampling,
        )
        self.encoded_dim = self.spatial_encoder.out_dim + self.time_encoder.out_dim
        self.model = self._build_torch_mlp(
            encoded_dim=self.encoded_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            output_activation=output_activation,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        编码输入坐标为特征

        Args:
            inputs: 输入坐标 (..., 3) 格式 [y, x, t]

        Returns:
            编码后的特征向量
        """
        spatial = inputs[..., :2]
        temporal = inputs[..., 2:3]
        encoded_parts = [
            self.spatial_encoder(spatial),
            self.time_encoder(temporal),
        ]
        return torch.cat(encoded_parts, dim=-1)

    def forward(self, encoded_inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            encoded_inputs: 编码后的输入或原始坐标

        Returns:
            预测的残差值
        """
        # Allow either pre-encoded features or raw (y,x,t) inputs.
        if encoded_inputs.shape[-1] == self.encoded_dim:
            encoded = encoded_inputs
        elif encoded_inputs.shape[-1] == 3:
            encoded = self.encode(encoded_inputs)
        else:
            raise ValueError(f"Expected input with last-dim == {self.encoded_dim} (encoded) or 3 (raw coords), got {encoded_inputs.shape[-1]}")
        return self.model(encoded)

    def reset_parameters(self) -> None:
        """重置模型参数"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def _build_torch_mlp(
        self,
        encoded_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        output_activation: str,
    ) -> nn.Module:
        """
        构建MLP网络

        Args:
            encoded_dim: 编码后特征维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            activation: 激活函数
            output_activation: 输出激活函数

        Returns:
            MLP网络
        """
        layers: List[nn.Module] = []
        in_features = encoded_dim
        n_hidden = max(1, num_layers)
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(self._activation_factory(activation))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        if output_activation == "Sigmoid":
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    @staticmethod
    def _activation_factory(name: str) -> nn.Module:
        """
        创建激活函数

        Args:
            name: 激活函数名称

        Returns:
            激活函数模块
        """
        if name == "ReLU":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation: {name}. Only 'ReLU' is supported now.")
