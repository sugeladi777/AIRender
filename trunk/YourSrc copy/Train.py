"""
Train.py: 光照贴图残差模型训练脚本

支持多GPU并行训练白天/夜晚模型，使用PyTorch进行优化。
包括数据加载、模型训练、评估和保存功能。
"""

from __future__ import annotations

import math
import random
import argparse
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from PIL import Image

from Model import Model
import Utils
from Dataset import LightmapDataLoader

@dataclass
class LightmapMetrics:
    psnr: float
    ssim: float
    lpips: float
    model_size_mb: float
    data_size_mb: float


class LightmapTrainer:
    """
    光照贴图残差模型训练器

    支持多GPU并行训练白天和夜晚模型，使用OneCycleLR调度器和torch.compile优化。
    封装了单个光照贴图的训练、评估和保存逻辑。
    """

    def __init__(self, args: argparse.Namespace, device: torch.device, data_loader: LightmapDataLoader):
        """初始化训练器"""
        self.args = args
        self.device = device
        self.data_loader = data_loader

    def build_model(
        self,
        hidden_dim: int,
        spatial_freq: int,
        time_freq: int,
        num_layers: int,
        activation: str,
        output_activation: str,
        use_compile: bool = True,
    ) -> Model:
        """
        构建并编译模型

        Args:
            hidden_dim: 隐藏层维度
            spatial_freq: 空间频率
            time_freq: 时间频率
            num_layers: 层数
            activation: 激活函数
            output_activation: 输出激活函数

        Returns:
            编译后的模型
        """
        model = Model(
            output_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            spatial_frequencies=spatial_freq,
            time_frequencies=time_freq,
            activation=activation,
            output_activation=output_activation,
        ).to(self.device)
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        if use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')
            except Exception as e:
                print(f"Warning: torch.compile failed, using uncompiled model: {e}")
        return model

    def train_one_lightmap(self, lightmap: Dict, times: List[str], night_times: List[str],
                           log_interval: int = 1000, no_save: bool = False
                           ) -> Tuple[Optional[LightmapMetrics], Optional[str]]:
        """
        训练单个光照贴图的白天和夜晚模型

        Args:
            lightmap: 光照贴图信息字典
            times: 时间列表
            night_times: 夜晚时间列表
            log_interval: 日志打印间隔
            no_save: 是否不保存结果

        Returns:
            评估指标和错误信息（如果失败）
        """
        try:
            resolution = lightmap['resolution']
            id_ = lightmap['id']
            time_count = len(times)

            coords_valid, segments_valid, residual_valid, baseline_day_flat, baseline_night_flat, baseline_per_time_tensor, mask_data, resolution = self.data_loader.prepare_training_tensors(
                lightmap,
                times,
                day_baseline_key="1200",
                night_baseline_key="0",
                night_time_keys=list(night_times),
                time_count=time_count,
            )

            # 支持白天/夜晚不同模型结构（若未提供则回退到通用参数）
            # day/night hidden sizes have defaults; read directly
            day_hidden = self.args.day_hidden_dim
            night_hidden = self.args.night_hidden_dim

            # spatial/time frequencies fallback to sensible constants when not provided
            day_spatial_freq = self.args.day_spatial_freq if getattr(self.args, 'day_spatial_freq', None) is not None else 8
            night_spatial_freq = self.args.night_spatial_freq if getattr(self.args, 'night_spatial_freq', None) is not None else 8

            day_time_freq = self.args.day_time_freq if getattr(self.args, 'day_time_freq', None) is not None else 4
            night_time_freq = self.args.night_time_freq if getattr(self.args, 'night_time_freq', None) is not None else 4

            day_num_layers = self.args.day_num_layers
            night_num_layers = self.args.night_num_layers

            # 模型在训练阶段按昼/夜分开创建与释放，降低峰值显存

            if self.args.loss_type == "l1":
                criterion = nn.L1Loss()
            elif self.args.loss_type == "l2":
                criterion = nn.MSELoss()
            else:
                criterion = nn.SmoothL1Loss()

            day_mask = segments_valid < 0.5
            night_mask = segments_valid >= 0.5

            # 训练白天/夜晚模型，允许不同的 iterations/lr/batch_size
            # day/night iterations/batch/lr 在 CLI 中已有默认值；直接读取即可
            day_iterations = self.args.day_iterations
            night_iterations = self.args.night_iterations

            day_lr = self.args.day_onecycle_max_lr if getattr(self.args, 'day_onecycle_max_lr', None) is not None else 1e-3
            night_lr = self.args.night_onecycle_max_lr if getattr(self.args, 'night_onecycle_max_lr', None) is not None else 1e-3

            # 推理/评估的 batch size 使用 day/night 中的较大者作为默认
            DEFAULT_BATCH = 25000
            day_batch = self.args.day_batch_size if getattr(self.args, 'day_batch_size', None) is not None else DEFAULT_BATCH
            night_batch = self.args.night_batch_size if getattr(self.args, 'night_batch_size', None) is not None else DEFAULT_BATCH

            # 依次训练白天与夜晚模型，训练完成即保存并释放
            day_model = None
            if day_mask.sum().item() > 0:
                day_model = self.build_model(
                    day_hidden,
                    day_spatial_freq,
                    day_time_freq,
                    day_num_layers,
                    self.args.activation,
                    self.args.output_activation,
                )
                day_coords = coords_valid[day_mask]
                day_targets = residual_valid[day_mask].to(self.device)
                self._train_segment_model(
                    model=day_model,
                    segment_name=f"day {lightmap['level']}_{id_}",
                    inputs=day_coords,
                    targets=day_targets,
                    criterion=criterion,
                    log_interval=log_interval,
                    iterations=day_iterations,
                    init_lr=day_lr,
                    batch_size=day_batch,
                )
                # 保存白天模型（包含 baseline_day）并释放
                self._save_single_model(
                    level=lightmap['level'],
                    lightmap_id=id_,
                    segment='day',
                    model=day_model,
                    baseline_flat=baseline_day_flat.detach().cpu().numpy(),
                )
                del day_coords, day_targets, day_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            night_model = None
            if night_mask.sum().item() > 0:
                night_model = self.build_model(
                    night_hidden,
                    night_spatial_freq,
                    night_time_freq,
                    night_num_layers,
                    self.args.activation,
                    self.args.output_activation,
                )
                night_coords = coords_valid[night_mask]
                night_targets = residual_valid[night_mask].to(self.device)
                self._train_segment_model(
                    model=night_model,
                    segment_name=f"night {lightmap['level']}_{id_}",
                    inputs=night_coords,
                    targets=night_targets,
                    criterion=criterion,
                    log_interval=log_interval,
                    iterations=night_iterations,
                    init_lr=night_lr,
                    batch_size=night_batch,
                )
                # 保存夜晚模型（包含 baseline_night）并释放
                self._save_single_model(
                    level=lightmap['level'],
                    lightmap_id=id_,
                    segment='night',
                    model=night_model,
                    baseline_flat=baseline_night_flat.detach().cpu().numpy(),
                )
                del night_coords, night_targets, night_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 评估阶段：重新构建模型并从已保存文件加载权重，避免训练阶段双模型并存
            day_model = self.build_model(
                day_hidden,
                day_spatial_freq,
                day_time_freq,
                day_num_layers,
                self.args.activation,
                self.args.output_activation,
            )
            night_model = self.build_model(
                night_hidden,
                night_spatial_freq,
                night_time_freq,
                night_num_layers,
                self.args.activation,
                self.args.output_activation,
            )
            self._load_model_weights_from_bin(day_model, f"./Parameters/model_{lightmap['level']}_{id_}_day_params.bin")
            self._load_model_weights_from_bin(night_model, f"./Parameters/model_{lightmap['level']}_{id_}_night_params.bin")

            metrics = self._evaluate_and_save(
                day_model,
                night_model,
                lightmap,
                times,
                night_times,
                resolution,
                baseline_per_time_tensor,
                mask_data,
                no_save=no_save,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics, None
        except Exception as ex:
            return None, str(ex)

    def _train_segment_model(
        self,
        model: Model,
        segment_name: str,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        log_interval: int,
        iterations: int,
        init_lr: float,
        batch_size: int,
    ) -> None:
        """
        训练单个模型段（白天或夜晚）

        使用OneCycleLR调度器和torch.compile优化训练过程。

        Args:
            model: 要训练的模型
            segment_name: 段名称（用于日志）
            inputs: 输入坐标
            targets: 目标残差
            criterion: 损失函数
            log_interval: 日志间隔
            iterations: 训练迭代次数
            init_lr: 初始学习率
            batch_size: 批大小
        """
        total_steps = max(1, iterations)
        init_lr = float(init_lr)
        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=init_lr,
            epochs=1,
            steps_per_epoch=iterations,
            pct_start=0.3,  # 30% of cycle for increasing LR
        )

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=0.0)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=10.0, neginf=-10.0)
        targets = torch.clamp(targets, -10.0, 10.0)  # Clip targets to prevent large gradients
        train_data = torch.cat([inputs, targets], dim=-1)
        del inputs, targets
        num_samples = train_data.shape[0]
        if num_samples == 0:
            return
        permutation_cpu = torch.randperm(num_samples, device='cpu')
        non_blocking = self.device.type == 'cuda'

        batch_start = 0
        pbar = trange(
            iterations,
            desc=f"Train {segment_name} (dev {self.device})",
            ncols=200,
        )
        for it in pbar:
            if batch_start >= num_samples:
                batch_start = 0
                permutation_cpu = torch.randperm(num_samples, device='cpu')
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = permutation_cpu[batch_start:batch_end].to(self.device, non_blocking=non_blocking)
            batch = train_data.index_select(0, batch_indices)
            inputs_batch = batch[:, :3]
            targets_batch = batch[:, 3:]

            optimizer.zero_grad()
            preds = model(inputs_batch)
            if torch.isnan(preds).any():
                print(f"NaN detected in predictions at iteration {it}, skipping batch")
                continue
            if targets_batch.dtype != preds.dtype:
                targets_batch = targets_batch.to(preds.dtype)
            loss = criterion(preds, targets_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            batch_start = batch_end

            if (
                (it + 1) % log_interval == 0
                or it == 0
                or (it + 1) == iterations
            ):
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{current_lr:.2e}")

        model.eval()

    def _evaluate_and_save(
        self,
        day_model: Model,
        night_model: Model,
        lightmap: Dict,
        times: List[str],
        night_times: List[str],
        resolution: Dict[str, int],
        baseline_per_time_tensor: torch.Tensor,
        mask_data: np.ndarray,
        no_save: bool = False,
    ) -> LightmapMetrics:
        """
        评估模型并保存结果

        Args:
            day_model: 白天模型
            night_model: 夜晚模型
            lightmap: 光照贴图信息
            times: 时间列表
            night_times: 夜晚时间列表
            resolution: 分辨率信息
            baseline_per_time_tensor: 基线张量
            mask_data: 遮罩数据
            no_save: 是否不保存

        Returns:
            评估指标
        """
        # 重新生成 coords_all 和 segment_per_sample 用于评估
        segment_flags = [1.0 if t in night_times else 0.0 for t in times]
        segment_flags_tensor = torch.tensor(segment_flags, dtype=torch.float32)
        coords_all, segment_per_sample = self.data_loader.prepare_coords(resolution, times, segment_flags_tensor)
        # 获取 residual_np_flat 和 baseline_per_time_flat
        residual_np_flat, _, _, _, baseline_per_time_flat, _ = self.data_loader.load_segmented_residuals(
            lightmap, times, day_baseline_key="1200", night_baseline_key="0", night_time_keys=list(night_times), time_count=len(times)
        )
        return evaluate_and_save(
            day_model,
            night_model,
            coords_all,
            segment_per_sample,
            baseline_per_time_tensor,
            residual_np_flat,
            baseline_per_time_flat,
            mask_data,
            resolution,
            times,
            lightmap['level'],
            lightmap['id'],
            max(getattr(self.args, 'day_batch_size', 25000), getattr(self.args, 'night_batch_size', 25000)),
            self.device,
            no_save=no_save,
        )

    @staticmethod
    def _save_single_model(
        level: str,
        lightmap_id: str,
        segment: str,
        model: Model,
        baseline_flat: np.ndarray,
    ) -> None:
        """
        保存单个模型参数到二进制文件

        Args:
            level: 关卡名称
            lightmap_id: 光照贴图ID
            segment: 段名称（day/night）
            model: 要保存的模型
            baseline_flat: 基线数据
        """
        params = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
        out_path = f"./Parameters/model_{level}_{lightmap_id}_{segment}_params.bin"
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        params.astype(np.float32).tofile(out_path)
        with open(out_path, 'ab') as f:
            baseline_flat.astype(np.float32, copy=False).tofile(f)

    @staticmethod
    def _load_model_weights_from_bin(model: Model, bin_path: str) -> None:
        """
        从二进制文件加载模型权重

        Args:
            model: 要加载权重的模型
            bin_path: 二进制文件路径
        """
        if not os.path.exists(bin_path):
            return
        params_array = np.fromfile(bin_path, dtype=np.float32)
        num_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            offset = 0
            for param in model.parameters():
                size = param.numel()
                if offset + size <= params_array.size:
                    param_data = params_array[offset:offset + size]
                    param.copy_(torch.from_numpy(param_data.reshape(param.shape)))
                else:
                    break
                offset += size



def build_model(
    hidden_dim: int,
    device: torch.device,
    spatial_freq: int,
    time_freq: int,
    num_layers: int,
    activation: str,
    output_activation: str,
) -> Model:
    """
    构建并编译模型（全局函数版本）

    Args:
        hidden_dim: 隐藏层维度
        device: 设备
        spatial_freq: 空间频率
        time_freq: 时间频率
        num_layers: 层数
        activation: 激活函数
        output_activation: 输出激活函数

    Returns:
        编译后的模型
    """
    model = Model(
        output_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        spatial_frequencies=spatial_freq,
        time_frequencies=time_freq,
        activation=activation,
        output_activation=output_activation,
    ).to(device)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    # Compile model for acceleration (PyTorch 2.0+)
    model = torch.compile(model, mode='default')
    return model


def safe_mean(values: List[float]) -> float:
    """
    安全计算平均值，避免空列表导致的错误

    Args:
        values: 数值列表

    Returns:
        平均值，如果列表为空则返回NaN
    """
    return float(np.mean(values)) if values else float('nan')


def evaluate_and_save(
    day_model: Model,
    night_model: Model,
    coords_all: torch.Tensor,
    segment_per_sample: torch.Tensor,
    baseline_per_time_tensor: torch.Tensor,
    residual_np_flat: np.ndarray,
    baseline_per_time_flat: np.ndarray,
    mask_data: np.ndarray,
    resolution: Dict[str, int],
    times: List[str],
    level: str,
    id_: str,
    batch_size: int,
    device: torch.device,
    no_save: bool = False,
) -> LightmapMetrics:
    """
    评估模型性能并保存结果图像

    计算PSNR、SSIM、LPIPS指标，分块处理大图像以节省显存。

    Args:
        day_model: 白天模型
        night_model: 夜晚模型
        coords_all: 所有坐标
        segment_per_sample: 段标签
        baseline_per_time_tensor: 基线张量
        residual_np_flat: 残差数据
        baseline_per_time_flat: 基线数据
        mask_data: 遮罩数据
        resolution: 分辨率
        times: 时间列表
        level: 关卡名称
        id_: 光照贴图ID
        batch_size: 批大小
        device: 设备
        no_save: 是否不保存图像

    Returns:
        评估指标
    """
    time_count = len(times)
    coords_cpu = coords_all
    segments_cpu = segment_per_sample
    if coords_cpu.shape[0] != segments_cpu.shape[0]:
        raise ValueError("Coordinate and segment label size mismatch during evaluation.")

    with torch.no_grad():
        day_model.eval()
        night_model.eval()

        pred_flat = torch.zeros((coords_cpu.shape[0], 3), dtype=torch.float32)

        day_indices = torch.nonzero(segments_cpu < 0.5, as_tuple=False).squeeze(-1)
        night_indices = torch.nonzero(segments_cpu >= 0.5, as_tuple=False).squeeze(-1)

        def infer_segment_chunked(model: Model, indices: torch.Tensor) -> None:
            if indices.numel() == 0:
                return
            for start in range(0, indices.numel(), batch_size):
                end = min(start + batch_size, indices.numel())
                chunk_positions = indices[start:end]
                # coords_cpu is CPU tensor; gather chunk on CPU then move to device
                chunk_coords = coords_cpu[chunk_positions]
                # move to device, run model (forward accepts raw coords)
                with torch.no_grad():
                    outputs = model(chunk_coords.to(device))
                pred_flat[chunk_positions] = outputs.detach().cpu()

        infer_segment_chunked(day_model, day_indices)
        infer_segment_chunked(night_model, night_indices)

        pred = pred_flat.reshape(time_count, resolution['height'], resolution['width'], 3)
        baseline_imgs = baseline_per_time_tensor.reshape(time_count, resolution['height'], resolution['width'], 3)
        torch.nan_to_num_(baseline_imgs, nan=0.0, posinf=1.0, neginf=0.0)
        pred = pred + baseline_imgs
        torch.nan_to_num_(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred = pred.permute(0, 3, 1, 2)

        gt_full = (
            residual_np_flat.reshape(time_count, resolution['height'] * resolution['width'], 3)
            + baseline_per_time_flat.reshape(time_count, resolution['height'] * resolution['width'], 3)
        )
        lightmap_data = torch.from_numpy(
            gt_full.reshape(time_count, resolution['height'], resolution['width'], 3)
        ).permute(0, 3, 1, 2)

        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []
        part_size = min(resolution['height'], resolution['width'])
        rows = (lightmap_data.shape[2] + part_size - 1) // part_size
        cols = (lightmap_data.shape[3] + part_size - 1) // part_size
        for t in range(time_count):
            pred[t, :, mask_data[t] <= 0] = 0
            for i in range(rows):
                for j in range(cols):
                    sr = i * part_size
                    er = min((i + 1) * part_size, lightmap_data.shape[2])
                    sc = j * part_size
                    ec = min((j + 1) * part_size, lightmap_data.shape[3])
                    lm_part = lightmap_data[[t], :, sr:er, sc:ec].to(device)
                    rec_part = pred[[t], :, sr:er, sc:ec].to(device)
                    mask_part = torch.from_numpy(mask_data[t, sr:er, sc:ec]).to(device)
                    valid_mask = mask_part >= 127
                    if not torch.any(valid_mask):
                        continue
                    if float(lm_part.abs().max().item()) == 0.0:
                        continue
                    psnr_list.append(Utils.cal_psnr(lm_part, rec_part, mask_part))
                    ssim_list.append(Utils.cal_ssim(lm_part, rec_part))
                    lpips_list.append(Utils.cal_lpips(lm_part, rec_part))

        # Filter out NaN/Inf values from metrics
        psnr_list = [p for p in psnr_list if math.isfinite(p)]
        ssim_list = [s for s in ssim_list if math.isfinite(s)]
        lpips_list = [l for l in lpips_list if math.isfinite(l)]

        psnr_mean = safe_mean(psnr_list)
        ssim_mean = safe_mean(ssim_list)
        lpips_mean = safe_mean(lpips_list)

        day_file = f"./Parameters/model_{level}_{id_}_day_params.bin"
        night_file = f"./Parameters/model_{level}_{id_}_night_params.bin"
        if os.path.exists(day_file) and os.path.exists(night_file):
            model_size_mb = (os.path.getsize(day_file) + os.path.getsize(night_file)) / 1024 / 1024
        else:
            baseline_bytes = baseline_per_time_flat.nbytes
            total_params = sum(p.numel() for p in day_model.parameters()) + sum(
                p.numel() for p in night_model.parameters()
            )
            model_size_mb = (total_params * 4 + baseline_bytes) / 1024 / 1024
        data_size_mb = lightmap_data.numel() * 4 / 1024 / 1024

        print(f"metrics of lightmap {level}_{id_}------------")
        print(f"PSNR: {psnr_mean}")
        print(f"SSIM: {ssim_mean}")
        print(f"LPIPS: {lpips_mean}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Data Size: {data_size_mb:.2f} MB")
        print("-----------------------------------------")

        if not no_save:
            pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            imgs = []
            for t in range(pred_np.shape[0]):
                im = np.nan_to_num(pred_np[t])
                im = np.clip(im, 0.0, 1.0)
                im8 = (im * 255.0).astype(np.uint8)
                imgs.append(im8)

            n = len(imgs)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            H, W = imgs[0].shape[0], imgs[0].shape[1]

            grid_h = rows * H
            grid_w = cols * W
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for idx, im in enumerate(imgs):
                r = idx // cols
                c = idx % cols
                sr = r * H
                sc = c * W
                grid[sr:sr+H, sc:sc+W, :] = im

            grid_img = Image.fromarray(grid)
            out_path = f'./ResultImages/grid_{level}_{id_}.png'
            grid_img.save(out_path)

        return LightmapMetrics(psnr_mean, ssim_mean, lpips_mean, model_size_mb, data_size_mb)

def parse_devices(dev_str: str) -> List[int]:
    """
    解析设备字符串为GPU ID列表

    Args:
        dev_str: 逗号分隔的设备ID字符串

    Returns:
        GPU ID列表
    """
    if not dev_str:
        return list(range(torch.cuda.device_count()))
    parts = [p.strip() for p in dev_str.split(',') if p.strip()]
    return [int(p) for p in parts]


def chunk_list(items: List, n_chunks: int) -> List[List]:
    """
    将列表分割为指定数量的块

    Args:
        items: 要分割的列表
        n_chunks: 块数量

    Returns:
        分割后的块列表
    """
    if n_chunks <= 0:
        return [items]
    chunks = [[] for _ in range(n_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % n_chunks].append(item)
    return chunks


def worker_process(device_id: int, lightmaps: List[Dict], args: argparse.Namespace,
                   times: List[str], night_times: List[str], data_loader: LightmapDataLoader, return_queue: mp.Queue):
    """
    多进程工作函数，在指定GPU上训练光照贴图

    Args:
        device_id: GPU设备ID
        lightmaps: 光照贴图列表
        args: 命令行参数
        times: 时间列表
        night_times: 夜晚时间列表
        data_loader: 数据加载器
        return_queue: 返回队列
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    trainer = LightmapTrainer(args, device, data_loader)
    local_metrics: List[Tuple[str, LightmapMetrics]] = []
    failed: List[str] = []
    for lm in lightmaps:
        print(f"[GPU {device_id}] Training lightmap {lm['level']}_{lm['id']}")
        metrics, reason = trainer.train_one_lightmap(
            lm, times, night_times,
            log_interval=args.log_interval,
            no_save=args.no_save,
        )
        if metrics is None:
            print(f"[GPU {device_id}] Failed: {lm['level']}_{lm['id']} reason={reason}")
            failed.append(f"{lm['level']}_{lm['id']}")
        else:
            local_metrics.append((f"{lm['level']}_{lm['id']}", metrics))
    return_queue.put((device_id, local_metrics, failed))


def main():
    """
    主函数：解析参数并启动多GPU训练

    支持单GPU和多GPU并行训练模式。
    """
    parser = argparse.ArgumentParser(description="Train lightmap residual model (multi-GPU capable)")
    parser.add_argument("--dataset", type=str, default='../Data/Data_HPRC')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=1000, help="打印 loss 的间隔步数")
    parser.add_argument("--save", dest="no_save", action="store_false", help="保存 EXR 文件（默认不保存）")
    parser.set_defaults(no_save=True)
    parser.add_argument("--activation", type=str, choices=["ReLU"], default="ReLU", help="隐层激活函数（仅支持 ReLU）")
    parser.add_argument("--output-activation", type=str, choices=["None", "Sigmoid"], default="None", help="输出激活：残差任务请选择 None")
    parser.add_argument("--loss-type", type=str, choices=["l1", "smooth_l1", "l2"], default="l1", help="主损失类型")
    parser.add_argument("--devices", type=str, default="", help="逗号分隔 GPU id 列表, 留空自动检测全部")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数(默认按 GPU 数), 可用于限制使用部分 GPU")
    parser.add_argument("--day-hidden-dim", type=int, default=256, help="白天模型隐藏宽度，默认 256")
    parser.add_argument("--night-hidden-dim", type=int, default=256, help="夜晚模型隐藏宽度，默认 256")
    parser.add_argument("--day-spatial-freq", type=int, default=12, help="白天空间频率，默认 6")
    parser.add_argument("--night-spatial-freq", type=int, default=12, help="夜晚空间频率，默认 6")
    parser.add_argument("--day-time-freq", type=int, default=4, help="白天时间频率，默认 4")
    parser.add_argument("--night-time-freq", type=int, default=4, help="夜晚时间频率，默认 4")
    parser.add_argument("--day-num-layers", type=int, default=5, help="白天隐藏层数，默认 5")
    parser.add_argument("--night-num-layers", type=int, default=5, help="夜晚隐藏层数，默认 5")
    parser.add_argument("--day-iterations", type=int, default=80000, help="白天训练步数，默认 80000")
    parser.add_argument("--night-iterations", type=int, default=80000, help="夜晚训练步数，默认 80000")
    parser.add_argument("--day-onecycle-max-lr", type=float, default=1e-3, help="白天 OneCycle 峰值学习率，默认 1e-4")
    parser.add_argument("--night-onecycle-max-lr", type=float, default=1e-3, help="夜晚 OneCycle 峰值学习率，默认 1e-4")
    parser.add_argument("--day-batch-size", type=int, default=25000, help="白天批大小，默认 25000")
    parser.add_argument("--night-batch-size", type=int, default=25000, help="夜晚批大小，默认 25000")
    parser.add_argument("--use-compile", action="store_true", default=False, help="启用torch.compile加速（默认关闭）")
    args = parser.parse_args()
    devices = parse_devices(args.devices)
    if args.workers > 0:
        devices = devices[:args.workers]
    print(f"Detected devices: {devices}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)
    metrics_path = os.path.join("./Parameters", "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(["timestamp", "lightmap", "psnr", "ssim", "lpips", "model_size_mb", "data_size_mb", "device"])
     
    times = LightmapDataLoader.default_times()
    night_times = LightmapDataLoader.get_night_times()
    data_loader = LightmapDataLoader(args.dataset)
    config = data_loader.config

    lightmaps = config['lightmap_list']
    if not devices or len(devices) <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = LightmapTrainer(args, device, data_loader)
        total_psnr: List[float] = []
        total_ssim: List[float] = []
        total_lpips: List[float] = []
        failed: List[str] = []
        for lm in lightmaps:
            print(f"[Single] Training lightmap {lm['level']}_{lm['id']}")
            metrics, reason = trainer.train_one_lightmap(
            lm, times, night_times,
            log_interval=args.log_interval,
            no_save=args.no_save,
        )
            if metrics is None:
                failed.append(f"{lm['level']}_{lm['id']}")
            else:
                with open(metrics_path, "a", newline='') as mf:
                    writer = csv.writer(mf)
                    writer.writerow([datetime.utcnow().isoformat(), f"{lm['level']}_{lm['id']}", metrics.psnr, metrics.ssim, metrics.lpips, metrics.model_size_mb, metrics.data_size_mb, str(device)])
                if not math.isnan(metrics.psnr):
                    total_psnr.append(metrics.psnr)
                if not math.isnan(metrics.ssim):
                    total_ssim.append(metrics.ssim)
                if not math.isnan(metrics.lpips):
                    total_lpips.append(metrics.lpips)
        if failed:
            print("Failed lightmaps:")
            for f in failed:
                print(f"  - {f}")
            print(f"Total failed: {len(failed)}")
            print("-----------------------------------------")
        print("metrics of total data set ---------------")
        print(f"PSNR of all lightmaps: {safe_mean(total_psnr)}")
        print(f"SSIM of all lightmaps: {safe_mean(total_ssim)}")
        print(f"LPIPS of all lightmaps: {safe_mean(total_lpips)}")
        print("-----------------------------------------")
        return

    chunks = chunk_list(lightmaps, len(devices))
    return_queue: mp.Queue = mp.Queue()
    processes = []
    for dev, subset in zip(devices, chunks):
        p = mp.Process(target=worker_process, args=(dev, subset, args, times, night_times, data_loader, return_queue))
        p.start()
        processes.append(p)
    collected = 0
    total_psnr: List[float] = []
    total_ssim: List[float] = []
    total_lpips: List[float] = []
    failed: List[str] = []
    while collected < len(processes):
        dev_id, mlist, flist = return_queue.get()
        for name, m in mlist:
            with open(metrics_path, "a", newline='') as mf:
                writer = csv.writer(mf)
                writer.writerow([datetime.utcnow().isoformat(), name, m.psnr, m.ssim, m.lpips, m.model_size_mb, m.data_size_mb, f"cuda:{dev_id}"])
            if not math.isnan(m.psnr):
                total_psnr.append(m.psnr)
            if not math.isnan(m.ssim):
                total_ssim.append(m.ssim)
            if not math.isnan(m.lpips):
                total_lpips.append(m.lpips)
        failed.extend(flist)
        collected += 1
        print(f"[Main] Collected results from GPU {dev_id}")
    for p in processes:
        p.join()

    if failed:
        print("Failed lightmaps:")
        for f in failed:
            print(f"  - {f}")
        print(f"Total failed: {len(failed)}")
        print("-----------------------------------------")
    print("metrics of total data set ---------------")
    print(f"PSNR of all lightmaps: {safe_mean(total_psnr)}")
    print(f"SSIM of all lightmaps: {safe_mean(total_ssim)}")
    print(f"LPIPS of all lightmaps: {safe_mean(total_lpips)}")
    print("-----------------------------------------")
    
if __name__ == "__main__":
    # Ensure multiprocessing uses 'fork' for Linux compatibility. If already set,
    # ignore the RuntimeError.
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass
    main()