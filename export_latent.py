from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from src.models.compressor import F1F2Model


def parse_args():
    """解析导出参数：指定检查点与导出目录。"""
    p = argparse.ArgumentParser(description='Export latent map and F2 model from checkpoint')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检查点（使用 map_location 以兼容 CPU/GPU）
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt['config']
    H, W = ckpt['image_size']
    time_feat_dim = ckpt.get('time_feat_dim', 2 * cfg.get('time_harmonics', 2))

    # 重建模型结构并加载权重
    model = F1F2Model(
        latent_dim=cfg['latent_dim'],
        f1_hidden=cfg['hidden_f1'],
        f1_layers=cfg['layers_f1'],
        f2_hidden=cfg['hidden_f2'],
        f2_layers=cfg['layers_f2'],
        time_feat_dim=time_feat_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 计算并保存整张 latent_map
    with torch.no_grad():
        latent = model.compute_latent_map(H, W, device=device)
    np.save(out_dir / 'latent_map.npy', latent.numpy())

    # 保存 F2 的状态字典和 TorchScript 可部署版本
    torch.save(model.f2.state_dict(), out_dir / 'f2_state_dict.pt')
    dummy_in = torch.randn(1, cfg['latent_dim'] + time_feat_dim, device=device)
    f2_ts = torch.jit.trace(model.f2.eval(), dummy_in)
    torch.jit.save(f2_ts, str(out_dir / 'f2_ts.pt'))

    print("Export completed: latent_map.npy, f2_state_dict.pt, f2_ts.pt")


if __name__ == '__main__':
    main()
