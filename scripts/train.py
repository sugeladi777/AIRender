from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.lightmap_dataset import LightMapTimeDataset
from src.models.delta_field import DeltaField, DeltaFieldConfig
from src.trainers.trainer import Trainer


def parse_args():
    """解析命令行参数并返回 args。

    主要参数包含数据路径、输出路径、训练超参与模型结构超参。
    """
    p = argparse.ArgumentParser(description='训练 DeltaField：直接学习 f(x,y,t)->ΔRGB（以 baseline 为基准的残差）')
    p.add_argument('--data_dir', type=str, required=True, help='包含 24 张图的文件夹，按字典序对应 t=0..23')
    p.add_argument('--out_dir', type=str, required=True, help='训练输出目录，用于保存检查点与导出结果')

    p.add_argument('--epochs', type=int, default=10, help='训练轮数（epoch）')
    p.add_argument('--batch_size', type=int, default=65536, help='每个 batch 的样本数；CPU 上请使用较小值（如 1024–4096）')
    p.add_argument('--lr', type=float, default=5e-4, help='学习率（Adam 默认）')

    # 模型结构参数
    p.add_argument('--hidden', type=int, default=64, help='SIREN 隐藏层宽度')
    p.add_argument('--layers', type=int, default=6, help='SIREN 隐藏层数量')

    p.add_argument('--time_harmonics', type=int, default=4, help='时间 Fourier 编码的频率 K（time feature 维度 = 2*K）')
    # XY/Time 编码
    p.add_argument('--xy_harmonics', type=int, default=4, help='坐标 (x,y) 的 Fourier 编码频率 K_xy；0 表示不使用')
    p.add_argument('--xy_include_input', action='store_true', help='在 XY 编码中是否包含原始 (x,y) 输入（默认不包含，开启后拼接原始坐标）')

    # 优化器与学习率策略
    p.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW 权重衰减（weight decay）')
    p.add_argument('--scheduler_patience', type=int, default=5, help='ReduceLROnPlateau 的 patience（以 epoch 为单位）')
    p.add_argument('--scheduler_factor', type=float, default=0.5, help='ReduceLROnPlateau 的 factor（乘法因子）')
    p.add_argument('--min_lr', type=float, default=1e-6, help='学习率下限，scheduler 不会降到更低')
    p.add_argument('--clip_grad', type=float, default=0.0, help='梯度裁剪阈值 (L2 norm)，<=0 则不裁剪')

    p.add_argument('--samples_per_epoch', type=int, default=None,
                   help='每个 epoch 随机采样的样本数；默认 = 全部像素×帧数（可能很大），可指定较小值以加速调试')
    p.add_argument('--num_workers', type=int, default=0, help='DataLoader 的子进程数（Windows 推荐 0 或 1）')
    p.add_argument('--seed', type=int, default=42, help='随机种子，用于复现')

    # 残差学习设置：以某一时刻图像为基准，学习时间变化的残差
    p.add_argument('--residual_mode', type=bool, default=True, help='启用残差训练：目标为 rgb - baseline(t=baseline_time)，默认始终启用')
    p.add_argument('--baseline_time', type=int, default=12, help='基准时间（整点小时），默认 12')

    p.add_argument('--save_every', type=int, default=10, help='每隔多少个 epoch 保存一次检查点')

    return p.parse_args()


def set_seed(seed: int):
    # 设置随机种子以便复现
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 如果使用 CUDA，启用 cudnn 的 benchmark（当输入尺寸稳定时可加速卷积等操作）
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 数据集与 DataLoader
    dataset = LightMapTimeDataset(
        image_dir=args.data_dir,
        time_harmonics=args.time_harmonics,
        sample_mode='random' if args.samples_per_epoch is not None else 'all',
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        residual_mode=args.residual_mode,
        baseline_time=args.baseline_time,
    )
    H, W = dataset.size

    # 若为残差模式，保存基准图像以便后续推理时加回
    if args.residual_mode:
        base_img = (np.clip(dataset.baseline_image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image
        Image.fromarray(base_img).save(Path(args.out_dir) / 'baseline.png')

    # 使用 persistent_workers 与 prefetch_factor 提升多进程 DataLoader 的性能（num_workers > 0 时）
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2,
    )

    # 构建 DeltaField 模型
    df_cfg = DeltaFieldConfig(
        time_harmonics=args.time_harmonics,
        xy_harmonics=args.xy_harmonics,
        xy_include_input=args.xy_include_input,
        hidden=args.hidden,
        layers=args.layers,
    )
    model = DeltaField(df_cfg).to(device)

    # 使用 AdamW 并启用 weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 使用 ReduceLROnPlateau 来根据训练损失自适应降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.min_lr)
    # 使用 L1 损失（相对于 MSE 更不偏向平滑，可保留边缘/细节）
    criterion = nn.L1Loss()

    # 使用封装的 Trainer 进行训练与保存
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        out_dir=out_dir,
        image_size=(H, W),
        save_every=args.save_every,
    )

    trainer.fit(loader=loader, start_epoch=1, epochs=args.epochs)


if __name__ == '__main__':
    main()
