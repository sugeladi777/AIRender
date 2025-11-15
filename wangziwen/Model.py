from typing import Literal, List

import math
import torch
import torch.nn as nn


# 保留简单实现，只使用 ReLU；不再包含 Sine/Snake 等自定义激活。


class _FourierFeatureEncoding(nn.Module):
    def __init__(
        self,
        in_dims: int,
        n_frequencies: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
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
        base = self.in_dims if self.include_input else 0
        return base + 2 * self.in_dims * self.n_frequencies

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
    """纯 PyTorch 版本的残差重建模型。

    输入约定: (y_norm, x_norm, t_norm, segment_flag)
    激活与输出类型可配置，默认输出为线性层结果（适合残差预测）。
    """

    def __init__(
        self,
        input_dim: int = 4,
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
        super().__init__()

        if input_dim != 4:
            raise ValueError("Model expects input_dim=4: (y_norm, x_norm, t_norm, segment_flag)")

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

        encoded_dim = self.spatial_encoder.out_dim + self.time_encoder.out_dim + 1
        self.model = self._build_torch_mlp(
            encoded_dim=encoded_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            output_activation=output_activation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != 4:
            raise ValueError("Expected inputs with 4 features: (y_norm, x_norm, t_norm, segment_flag)")

        spatial = inputs[..., :2]
        temporal = inputs[..., 2:3]
        segment = inputs[..., 3:4]
        encoded = torch.cat(
            (
                self.spatial_encoder(spatial),
                self.time_encoder(temporal),
                segment,
            ),
            dim=-1,
        )
        return self.model(encoded)

    def reset_parameters(self) -> None:
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
        if name == "ReLU":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation: {name}. Only 'ReLU' is supported now.")
