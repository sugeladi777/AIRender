from __future__ import annotations

import math
import random
import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
# Do not set start method at import time; set it in the __main__ guard instead
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
from ReadData import (
    load_dataset_config,
    load_segmented_residuals,
    default_times,
    get_night_times,
)

def parse_times(time_str: str) -> float:
    # 将 0,100,... 转为小时比例（0.00,1.00,... -> /24 用于时间归一化）
    return int(time_str) / 100.0


@dataclass
class LightmapMetrics:
    psnr: float
    ssim: float
    lpips: float
    model_size_mb: float
    data_size_mb: float


class LightmapTrainer:
    """Encapsulates per-lightmap training & evaluation for clarity."""
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device

    def build_model(self, hidden_dim: int, spatial_freq: int, time_freq: int,
                    num_layers: int, activation: str, output_activation: str) -> Model:
        model = Model(
            input_dim=4,
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
        return model

    @staticmethod
    def prepare_coords(resolution: Dict[str, int], times: List[str], segment_flags_tensor: torch.Tensor) -> torch.Tensor:
        H, W = resolution['height'], resolution['width']
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        coords_np = np.stack([ys / (H - 1), xs / (W - 1)], axis=-1).reshape(-1, 2)
        base_coords = torch.tensor(coords_np, dtype=torch.float32)
        coords_all = []
        for time_idx, t_key in enumerate(times):
            alpha = torch.full((H * W, 1), parse_times(t_key) / 24, dtype=torch.float32)
            segment_flag = torch.full((H * W, 1), float(segment_flags_tensor[time_idx].item()), dtype=torch.float32)
            coords_with_time = torch.cat([base_coords, alpha, segment_flag], dim=-1)
            coords_all.append(coords_with_time)
        return torch.cat(coords_all, dim=0)

    def train_one_lightmap(self, lightmap: Dict, times: List[str], night_times: List[str],
                           log_interval: int = 1000, resume_mode: str = "off", no_save: bool = False
                           ) -> Tuple[Optional[LightmapMetrics], Optional[str]]:
        try:
            resolution = lightmap['resolution']
            id_ = lightmap['id']
            time_count = len(times)

            residual_np_flat, mask_data, baseline_day_flat, baseline_night_flat, baseline_per_time_flat, segment_flags = load_segmented_residuals(
                self.args.dataset,
                lightmap,
                times,
                day_baseline_key="1200",
                night_baseline_key="0",
                night_time_keys=list(night_times),
                time_count=time_count,
            )

            residual_data = torch.tensor(residual_np_flat, dtype=torch.float32)
            baseline_per_time_tensor = torch.tensor(baseline_per_time_flat, dtype=torch.float32)
            segment_flags_tensor = torch.tensor(segment_flags, dtype=torch.float32)

            model = self.build_model(
                self.args.hidden_dim,
                self.args.spatial_freq,
                self.args.time_freq,
                self.args.num_layers,
                self.args.activation,
                self.args.output_activation,
            )

            param_path = f"./Parameters/model_{lightmap['level']}_{id_}_params.bin"
            if resume_mode in {"skip", "continue"} and os.path.exists(param_path):
                print(f"[Resume] Found existing param file: {param_path}")
                arr = np.fromfile(param_path, dtype=np.float32)
                num_params = sum(p.numel() for p in model.parameters())
                if arr.size >= num_params:
                    offset = 0
                    with torch.no_grad():
                        for p in model.parameters():
                            size = p.numel()
                            pdata = arr[offset:offset+size]
                            if pdata.size != size:
                                break
                            p.copy_(torch.from_numpy(pdata.reshape(p.shape)))
                            offset += size
                else:
                    print(f"[Resume Warning] Saved parameters too small, ignoring: {param_path}")
                if resume_mode == "skip":
                    print(f"[Resume] Skip training for {lightmap['level']}_{id_}, evaluate only.")
                    metrics = evaluate_and_save(
                        model,
                        self.prepare_coords(resolution, times, torch.tensor(segment_flags, dtype=torch.float32)),
                        torch.tensor(baseline_per_time_flat, dtype=torch.float32),
                        residual_np_flat,
                        baseline_per_time_flat,
                        mask_data,
                        resolution,
                        times,
                        lightmap['level'],
                        id_,
                        self.args.batch_size,
                        self.device,
                        no_save=no_save,
                    )
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return metrics, None
                else:
                    print(f"[Resume] Continue training for {lightmap['level']}_{id_}.")

            if self.args.loss_type == "l1":
                criterion = nn.L1Loss()
            elif self.args.loss_type == "l2":
                criterion = nn.MSELoss()
            else:
                criterion = nn.SmoothL1Loss()

            total_steps = max(1, self.args.iterations)
            div_factor = float(self.args.onecycle_div_factor)
            final_div_factor = float(self.args.onecycle_final_div_factor)
            pct_start = float(self.args.onecycle_pct_start)
            init_lr = self.args.onecycle_max_lr / div_factor
            optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=self.args.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.onecycle_max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )
            try:
                print(f"OneCycleLR: initial_lr={init_lr:.2e}, peak_lr={self.args.onecycle_max_lr:.2e}, final_lr={(self.args.onecycle_max_lr / final_div_factor):.2e}, pct_start={pct_start}")
            except Exception:
                pass

            all_coords = self.prepare_coords(resolution, times, segment_flags_tensor)
            mask_flat = torch.from_numpy(mask_data.reshape(-1))
            valid_mask = (mask_flat >= 127)
            if valid_mask.numel() != all_coords.shape[0]:
                raise ValueError("Mask/coord length mismatch after reshape.")
            train_coords = all_coords[valid_mask]
            train_residuals = residual_data[valid_mask]
            if train_coords.numel() == 0:
                raise ValueError("No valid samples after mask filtering.")

            train_data = torch.cat([train_coords, train_residuals], dim=-1).to(self.device)
            permutation = torch.randperm(train_data.shape[0], device=self.device)
            train_data = train_data[permutation]
            num_samples = train_data.shape[0]

            batch_start = 0
            pbar = trange(self.args.iterations, desc=f"Train {lightmap['level']}_{id_} (dev {self.device})", ncols=110)
            for it in pbar:
                batch_end = min(batch_start + self.args.batch_size, num_samples)
                batch_data = train_data[batch_start:batch_end]
                target = batch_data[:, 4:]
                optimizer.zero_grad()
                pred = model(batch_data[:, :4])
                if target.dtype != pred.dtype:
                    target = target.to(pred.dtype)
                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                batch_start = batch_end
                if batch_start >= num_samples:
                    batch_start = 0
                    permutation = torch.randperm(train_data.shape[0], device=self.device)
                    train_data = train_data[permutation]
                    num_samples = train_data.shape[0]

                if (it + 1) % log_interval == 0 or it == 0 or (it + 1) == self.args.iterations:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{current_lr:.2e}")

            # Save parameters (model + baselines)
            all_params = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
            params_array = np.concatenate(all_params)
            out_path = f"./Parameters/model_{lightmap['level']}_{id_}_params.bin"
            params_array.astype(np.float32).tofile(out_path)
            with open(out_path, 'ab') as f:
                baseline_day_flat.astype(np.float32).tofile(f)
                baseline_night_flat.astype(np.float32).tofile(f)

            metrics = evaluate_and_save(
                model,
                all_coords,
                baseline_per_time_tensor,
                residual_np_flat,
                baseline_per_time_flat,
                mask_data,
                resolution,
                times,
                lightmap['level'],
                id_,
                self.args.batch_size,
                self.device,
                no_save=no_save,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics, None
        except Exception as ex:
            return None, str(ex)



def build_model(
    hidden_dim: int,
    device: torch.device,
    spatial_freq: int,
    time_freq: int,
    num_layers: int,
    activation: str,
    output_activation: str,
) -> Model:
    model = Model(
        input_dim=4,
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
    return model


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def prepare_coords(resolution: Dict[str, int], times: List[str], segment_flags_tensor: torch.Tensor) -> torch.Tensor:
    H, W = resolution['height'], resolution['width']
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    coords_np = np.stack([ys / (H - 1), xs / (W - 1)], axis=-1).reshape(-1, 2)
    base_coords = torch.tensor(coords_np, dtype=torch.float32)

    coords_all = []
    for time_idx, t_key in enumerate(times):
        alpha = torch.full((H * W, 1), parse_times(t_key) / 24, dtype=torch.float32)
        segment_flag = torch.full((H * W, 1), float(segment_flags_tensor[time_idx].item()), dtype=torch.float32)
        coords_with_time = torch.cat([base_coords, alpha, segment_flag], dim=-1)
        coords_all.append(coords_with_time)
    return torch.cat(coords_all, dim=0)


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float('nan')




def evaluate_and_save(
    model: Model,
    all_coords: torch.Tensor,
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
    time_count = len(times)
    with torch.no_grad():
        model.eval()
        pred_list = []
        for i in range((all_coords.shape[0] + batch_size - 1) // batch_size):
            s = i * batch_size
            e = min(s + batch_size, all_coords.shape[0])
            batch_data = all_coords[s:e].to(device)
            pred = model(batch_data[:, :4])
            pred_list.append(pred.cpu())
        pred = torch.cat(pred_list, dim=0)
        pred = pred.reshape(time_count, resolution['height'], resolution['width'], 3)
        baseline_imgs = baseline_per_time_tensor.reshape(time_count, resolution['height'], resolution['width'], 3)
        pred = pred + baseline_imgs
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
        part_size = 256
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
                    if torch.any(valid_mask) and lm_part.max() != 0:
                        psnr_list.append(Utils.cal_psnr(lm_part, rec_part, mask_part))
                        ssim_list.append(Utils.cal_ssim(lm_part, rec_part))
                        lpips_list.append(Utils.cal_lpips(lm_part, rec_part))

        psnr_mean = safe_mean(psnr_list)
        ssim_mean = safe_mean(ssim_list)
        lpips_mean = safe_mean(lpips_list)

        model_file = f"./Parameters/model_{level}_{id_}_params.bin"
        if os.path.exists(model_file):
            model_size_mb = os.path.getsize(model_file) / 1024 / 1024
        else:
            # fallback: estimate from parameter count + baseline bytes
            baseline_bytes = baseline_per_time_flat.nbytes  # 已包含所有时间的基准图
            model_size_mb = (sum(p.numel() for p in model.parameters()) * 4 + baseline_bytes) / 1024 / 1024
        data_size_mb = lightmap_data.numel() * 4 / 1024 / 1024

        print(f"metrics of lightmap {level}_{id_}------------")
        print(f"PSNR: {psnr_mean}")
        print(f"SSIM: {ssim_mean}")
        print(f"LPIPS: {lpips_mean}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Data Size: {data_size_mb:.2f} MB")
        print("-----------------------------------------")

        # 保存为一张网格图（可选）——不再保存为单张 EXR
        if not no_save:
            # pred: (time_count, 3, H, W) -> (time_count, H, W, 3)
            pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)

            # 简单线性归一并转换为 8 位 RGB 用于保存（保持比例关系）
            # 对每帧裁剪到 [0,1]，然后映射到 [0,255]
            imgs = []
            for t in range(pred_np.shape[0]):
                im = np.nan_to_num(pred_np[t])
                im = np.clip(im, 0.0, 1.0)
                im8 = (im * 255.0).astype(np.uint8)
                imgs.append(im8)

            # 构建网格：cols x rows
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
    if not dev_str:
        return list(range(torch.cuda.device_count()))
    parts = [p.strip() for p in dev_str.split(',') if p.strip()]
    return [int(p) for p in parts]


def chunk_list(items: List, n_chunks: int) -> List[List]:
    if n_chunks <= 0:
        return [items]
    chunks = [[] for _ in range(n_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % n_chunks].append(item)
    return chunks


def worker_process(device_id: int, lightmaps: List[Dict], args: argparse.Namespace,
                   times: List[str], night_times: List[str], return_queue: mp.Queue):
    # Only set CUDA device if CUDA is available. Users may pass --devices manually
    # even when CUDA is not present; avoid calling set_device in that case.
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device_id)
        except Exception:
            # fallback: continue without setting device index explicitly
            pass
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    trainer = LightmapTrainer(args, device)
    # store tuples of (lightmap_name, metrics)
    local_metrics: List[Tuple[str, LightmapMetrics]] = []
    failed: List[str] = []
    for lm in lightmaps:
        print(f"[GPU {device_id}] Training lightmap {lm['level']}_{lm['id']}")
        metrics, reason = trainer.train_one_lightmap(
            lm, times, night_times,
            log_interval=args.log_interval,
            resume_mode=args.resume,
            no_save=args.no_save,
        )
        if metrics is None:
            print(f"[GPU {device_id}] Failed: {lm['level']}_{lm['id']} reason={reason}")
            failed.append(f"{lm['level']}_{lm['id']}")
        else:
            local_metrics.append((f"{lm['level']}_{lm['id']}", metrics))
    return_queue.put((device_id, local_metrics, failed))


def main():
    parser = argparse.ArgumentParser(description="Train lightmap residual model (multi-GPU capable)")
    parser.add_argument("--iterations", type=int, default=80000)
    parser.add_argument("--batch_size", type=int, default=25000)
    parser.add_argument("--dataset", type=str, default='../Data/Data_HPRC')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10000, help="打印 loss 的间隔步数")
    parser.add_argument("--resume", type=str, choices=["off", "skip", "continue"], default="off", help="断点机制：off=正常训练，skip=已存在则跳过，continue=载入后继续训练")
    # 默认不保存 EXR 文件；传入 --save 可开启保存
    parser.add_argument("--save", dest="no_save", action="store_false", help="保存 EXR 文件（默认不保存）")
    parser.set_defaults(no_save=True)
    # 网络结构与损失可配置
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--spatial-freq", type=int, default=8, help="空间频率编码数量 (y,x)")
    parser.add_argument("--time-freq", type=int, default=4, help="时间维频率编码数量 (t)")
    parser.add_argument("--num-layers", type=int, default=5, help="隐藏层层数 (tcnn: n_hidden_layers)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--activation", type=str, choices=["ReLU"], default="ReLU", help="隐层激活函数（仅支持 ReLU）")
    parser.add_argument("--output-activation", type=str, choices=["None", "Sigmoid"], default="None", help="输出激活：残差任务请选择 None")
    parser.add_argument("--loss-type", type=str, choices=["l1", "smooth_l1", "l2"], default="l1", help="主损失类型")
    # OneCycleLR
    parser.add_argument("--onecycle-max-lr", type=float, default=1e-3, help="OneCycleLR 的峰值学习率")
    parser.add_argument("--onecycle-div-factor", type=float, default=5.0, help="OneCycleLR 的 div_factor（初始 lr = max_lr / div_factor）")
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1e2, help="OneCycleLR 的 final_div_factor（最终 lr = max_lr / final_div_factor）")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.2, help="OneCycleLR 的 pct_start（上升阶段比例）")
    parser.add_argument("--devices", type=str, default="", help="逗号分隔 GPU id 列表, 留空自动检测全部")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数(默认按 GPU 数), 可用于限制使用部分 GPU")
    args = parser.parse_args()
    devices = parse_devices(args.devices)
    if args.workers > 0:
        devices = devices[:args.workers]
    print(f"Detected devices: {devices}")

    # 全局随机种子，确保每个运行可重复；每个场景会在此基础上再偏移
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建保存参数和结果的文件夹
    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)
    # metrics output file (main process will write to it)
    metrics_path = os.path.join("./Parameters", "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(["timestamp", "lightmap", "psnr", "ssim", "lpips", "model_size_mb", "data_size_mb", "device"])
     
    # 读取数据集配置与时间列表（抽离）
    times = default_times()
    night_times = get_night_times()
    config = load_dataset_config(args.dataset)

    lightmaps = config['lightmap_list']
    if not devices or len(devices) <= 1:
        # 单 GPU 或无 GPU 情况：直接顺序执行
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = LightmapTrainer(args, device)
        total_psnr: List[float] = []
        total_ssim: List[float] = []
        total_lpips: List[float] = []
        failed: List[str] = []
        for lm in lightmaps:
            print(f"[Single] Training lightmap {lm['level']}_{lm['id']}")
            metrics, reason = trainer.train_one_lightmap(
                lm, times, night_times,
                log_interval=args.log_interval,
                resume_mode=args.resume,
                no_save=args.no_save,
            )
            if metrics is None:
                failed.append(f"{lm['level']}_{lm['id']}")
            else:
                # write per-lightmap metrics to CSV
                with open(metrics_path, "a", newline='') as mf:
                    writer = csv.writer(mf)
                    writer.writerow([datetime.utcnow().isoformat(), f"{lm['level']}_{lm['id']}", metrics.psnr, metrics.ssim, metrics.lpips, metrics.model_size_mb, metrics.data_size_mb, str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))])
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

    # 多 GPU 情况：分块并行
    chunks = chunk_list(lightmaps, len(devices))
    return_queue: mp.Queue = mp.Queue()
    processes = []
    for dev, subset in zip(devices, chunks):
        p = mp.Process(target=worker_process, args=(dev, subset, args, times, night_times, return_queue))
        p.start()
        processes.append(p)
    # 收集结果
    collected = 0
    total_psnr: List[float] = []
    total_ssim: List[float] = []
    total_lpips: List[float] = []
    failed: List[str] = []
    while collected < len(processes):
        dev_id, mlist, flist = return_queue.get()
        # mlist contains tuples (lightmap_name, metrics)
        for name, m in mlist:
            # write per-lightmap metrics to CSV (main process does I/O)
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
    # Ensure multiprocessing uses 'spawn' for CUDA safety. If already set,
    # ignore the RuntimeError.
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()