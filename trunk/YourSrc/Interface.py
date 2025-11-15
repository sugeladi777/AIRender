import os
import numpy as np
import torch

from Model import LightmapModel

class BasicInterface:
    """简易推理接口，用于加载并推理单张lightmap"""
    def __init__(self, lightmap_config, device, param_dir: str = "./Parameters"):
        self.device = device
        self.param_dir = param_dir
        resolution = lightmap_config['resolution']
        self.model = LightmapModel(
            hidden_dim=16,
            feature_width=resolution['width'],
            feature_height=resolution['height'],
        )

        # 加载模型参数
        level = lightmap_config['level']
        lm_id = lightmap_config['id']
        param_path = os.path.join(self.param_dir, f"model_{level}_{lm_id}_params.bin")
        params_array = np.fromfile(param_path, dtype=np.float32)
        param_idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                param_size = param.numel()
                param_data = params_array[param_idx:param_idx + param_size]
                param.data = torch.from_numpy(param_data.reshape(param.shape))
                param_idx += param_size
        self.model.eval()
        self.model.to(self.device)
        
        # 初始化输入坐标
        self.resolution = resolution
        self.height = resolution['height']
        self.width = resolution['width']
        self.time_denominator = float(lightmap_config.get('time_count', 24))

    def reconstruct(self, current_time):
        H, W = self.height, self.width
        ys, xs = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        self.coords = torch.stack([
            ys.ravel() / (H - 1),
            xs.ravel() / (W - 1),
            torch.full((H * W,), float(current_time), device=self.device) / max(1.0, self.time_denominator)
        ], dim=-1).float()

        self.result = self.model(self.coords)
    
    def get_result(self):
        return self.result.reshape(self.resolution['height'], self.resolution['width'], 3).permute(2, 0, 1).unsqueeze(0)
    
    def random_test(self, coord):
        # 处理输入坐标
        coord[:, 0] = coord[:, 0] / (self.resolution['height'] - 1)
        coord[:, 1] = coord[:, 1] / (self.resolution['width'] - 1)
        coord[:, 2] = coord[:, 2] / max(1.0, self.time_denominator)
        coord = torch.from_numpy(coord).to(torch.float32).to(self.device)
        result = self.model(coord)
        return result


def get(lightmap_config, device):
    return BasicInterface(lightmap_config, device)
