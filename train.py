from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.lightmap_dataset import LightMapTimeDataset
from src.models.compressor import F1F2Model


def parse_args():
    """解析命令行参数并返回 args。

    主要参数包含数据路径、输出路径、训练超参与模型结构超参。
    """
    p = argparse.ArgumentParser(description='训练 F1+F2 SIREN 光照压缩模型（24 帧）')
    p.add_argument('--data_dir', type=str, required=True, help='包含 24 张图的文件夹，按字典序对应 t=0..23')
    p.add_argument('--out_dir', type=str, required=True, help='训练输出目录，用于保存检查点与导出结果')

    p.add_argument('--epochs', type=int, default=200, help='训练轮数（epoch）')
    p.add_argument('--batch_size', type=int, default=65536, help='每个 batch 的样本数；CPU 上请使用较小值（如 1024–4096）')
    p.add_argument('--lr', type=float, default=1e-3, help='学习率（Adam 默认）')

    # 模型结构参数，已与 compressor 默认值对齐
    p.add_argument('--latent_dim', type=int, default=32, help='潜变量维度 z 的大小（latent map 每像素向量长度）')
    p.add_argument('--hidden_f1', type=int, default=32, help='F1 隐藏层宽度（SIREN 隐藏单元数）')
    p.add_argument('--layers_f1', type=int, default=4, help='F1 的 SIREN 层数')
    p.add_argument('--hidden_f2', type=int, default=64, help='F2 隐藏层宽度（SIREN 隐藏单元数）')
    p.add_argument('--layers_f2', type=int, default=4, help='F2 的 SIREN 层数')

    p.add_argument('--time_harmonics', type=int, default=2, help='时间 Fourier 编码的频率 K（time feature 维度 = 2*K）')
    # 新增：XY 位置编码（Fourier 特征）
    p.add_argument('--xy_harmonics', type=int, default=0, help='坐标 (x,y) 的 Fourier 编码频率 K_xy；0 表示不使用')
    p.add_argument('--xy_include_input', action='store_true', help='在 XY 编码中是否包含原始 (x,y) 输入（默认不包含，开启后拼接原始坐标）')

    p.add_argument('--samples_per_epoch', type=int, default=None,
                   help='每个 epoch 随机采样的样本数；默认 = 全部像素×帧数（可能很大），可指定较小值以加速调试')
    p.add_argument('--num_workers', type=int, default=0, help='DataLoader 的子进程数（Windows 推荐 0 或 1）')
    p.add_argument('--seed', type=int, default=42, help='随机种子，用于复现')

    p.add_argument('--save_every', type=int, default=10, help='每隔多少个 epoch 保存一次检查点')
    p.add_argument('--compute_latent', action='store_true', help='训练结束后计算并保存 latent_map.npy 与 F2 权重')

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

    # 自动选择设备（可通过命令行在以后扩展为强制 CPU 或指定 cuda:0）
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
    )
    H, W = dataset.size

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

    time_feat_dim = 2 * args.time_harmonics

    # 构建模型并移动到目标设备
    model = F1F2Model(
        latent_dim=args.latent_dim,
        f1_hidden=args.hidden_f1,
        f1_layers=args.layers_f1,
        f2_hidden=args.hidden_f2,
        f2_layers=args.layers_f2,
        time_feat_dim=time_feat_dim,
        xy_harmonics=args.xy_harmonics,
        xy_include_input=args.xy_include_input,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 不使用混合精度（AMP）——保持数值稳定性，使用标准训练流程
    scaler = None

    best_loss: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        count = 0
        for xy, t_feat, rgb in pbar:
            # 将数据移动到 device，若使用 pin_memory 可使用 non_blocking 加速拷贝
            non_block = True if device.type == 'cuda' else False
            xy = xy.to(device, non_blocking=non_block)
            t_feat = t_feat.to(device, non_blocking=non_block)
            rgb = rgb.to(device, non_blocking=non_block)

            pred, _ = model(xy, t_feat)
            loss = criterion(pred, rgb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = xy.shape[0]
            running_loss += loss.item() * bs
            count += bs
            pbar.set_postfix(loss=running_loss / count)

        epoch_loss = running_loss / max(1, count)

        # 周期性保存检查点
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': vars(args),
                'image_size': (H, W),
                'time_feat_dim': time_feat_dim,
            }
            torch.save(ckpt, out_dir / 'last.ckpt')

        # 保存最优模型
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': vars(args),
                'image_size': (H, W),
                'time_feat_dim': time_feat_dim,
            }, out_dir / 'best.ckpt')

    # 训练结束后根据需要导出 latent map 与 F2
    if args.compute_latent:
        print("Computing latent map with best checkpoint...")
        # 重新加载最佳模型并计算整个图的 latent_map
        best = torch.load(out_dir / 'best.ckpt', map_location=device)
        model.load_state_dict(best['model'])
        model.eval()
        with torch.no_grad():
            latent = model.compute_latent_map(H, W, device=device)
        np.save(out_dir / 'latent_map.npy', latent.numpy())
        # 保存 F2 的权重与 TorchScript 版本以便部署
        f2_state = model.f2.state_dict()
        torch.save(f2_state, out_dir / 'f2_state_dict.pt')
        dummy_in = torch.randn(1, args.latent_dim + time_feat_dim, device=device)
        f2_ts = torch.jit.trace(model.f2.eval(), dummy_in)
        torch.jit.save(f2_ts, str(out_dir / 'f2_ts.pt'))
        print("Exported latent_map.npy, f2_state_dict.pt, f2_ts.pt")


if __name__ == '__main__':
    main()
