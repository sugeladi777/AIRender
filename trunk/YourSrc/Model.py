import torch
import torch.nn as nn
import torch.nn.functional as F


class LightmapModel(nn.Module):
    """简易lightmap网络，叠加MLP与可学习特征图"""

    def __init__(
        self,
        coord_dim: int = 3,
        output_dim: int = 3,
        hidden_dim: int = 256,
        feature_channels: int = 4,
        feature_width: int = 32,
        feature_height: int = 32,
    ) -> None:
        super().__init__()
        input_dim = coord_dim + feature_channels
        self.featuremap = nn.Parameter(
            torch.empty((1, feature_channels, feature_width, feature_height), dtype=torch.float32)
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        with torch.no_grad():
            nn.init.normal_(self.featuremap, mean=0.0, std=0.1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords包含归一化后的(y, x, t)
        grid = coords[:, :2].unsqueeze(0).unsqueeze(2) * 2.0 - 1.0
        feature = F.grid_sample(
            self.featuremap,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        feature = feature.squeeze(0).squeeze(-1).permute(1, 0)
        return self.mlp(torch.cat([coords, feature], dim=-1))
