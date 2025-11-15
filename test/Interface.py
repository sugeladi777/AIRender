import torch
import os
import numpy as np

from ExampleModel import ExampleModel

class BasicInterface:
    def __init__(self, lightmap_config, device):
        self.device = device
        self.model = ExampleModel(input_dim=7, output_dim=3, hidden_dim=16, feature_width=lightmap_config['resolution']['width'], feature_height=lightmap_config['resolution']['height'])

        # 加载模型参数
        level = lightmap_config['level']
        id = lightmap_config['id']
        params_array = np.fromfile(f"./Parameters/HPRC/Parameters-normal/model_{level}_{id}_params.bin", dtype=np.float32)
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
        resolution = lightmap_config['resolution']
        self.resolution = resolution
        self.height = resolution['height']
        self.width = resolution['width']

    def reconstruct(self, current_time):
        H, W = self.height, self.width
        ys, xs = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        self.coords = torch.stack([
            ys.ravel() / (H - 1),
            xs.ravel() / (W - 1),
            torch.full((H * W,), float(current_time), device=self.device) / 24.0
        ], dim=-1).float()

        self.result = self.model(self.coords)
    
    def get_result(self):
        return self.result.reshape(self.resolution['height'], self.resolution['width'], 3).permute(2, 0, 1).unsqueeze(0)
    
    def random_test(self, coord):
        # 处理输入坐标
        coord[:, 0] = coord[:, 0] / (self.resolution['height'] - 1)
        coord[:, 1] = coord[:, 1] / (self.resolution['width'] - 1)
        coord[:, 2] = (coord[:, 2]) / 24.0
        coord = torch.from_numpy(coord).to(torch.float32).to(self.device)
        result = self.model(coord)
        return result


def get(lightmap_config, device):
    return BasicInterface(lightmap_config, device)
