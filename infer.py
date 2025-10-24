from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.models.compressor import F1F2Model
from src.utils.encoding import encode_time


def parse_args():
    """解析推理参数：支持从 ckpt 或 latent+f2 加载并渲染指定时间的图像。"""
    p = argparse.ArgumentParser(description='Reconstruct image at time t using latent map + F2')
    p.add_argument('--latent_path', type=str, default=None, help='latent_map.npy')
    p.add_argument('--f2_path', type=str, default=None, help='f2_state_dict.pt')
    p.add_argument('--ckpt', type=str, default=None, help='Alternative: use combined checkpoint .ckpt')

    p.add_argument('--time', type=float, required=True, help='time in [0,24]')
    p.add_argument('--time_harmonics', type=int, default=None, help='override time harmonics (if using latent+f2)')
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--out', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 校验输入
    if args.ckpt is None and (args.latent_path is None or args.f2_path is None):
        raise ValueError('Provide either --ckpt or both --latent_path and --f2_path')

    if args.ckpt is not None:
        # 如果传入 ckpt，则用模型一次性计算 latent_map
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ckpt['config']
        H, W = ckpt['image_size']
        time_feat_dim = ckpt.get('time_feat_dim', 2 * cfg.get('time_harmonics', 2))
        xy_harmonics = cfg.get('xy_harmonics', 0)
        xy_include_input = cfg.get('xy_include_input', True)

        model = F1F2Model(
            latent_dim=cfg['latent_dim'],
            f1_hidden=cfg['hidden_f1'],
            f1_layers=cfg['layers_f1'],
            f2_hidden=cfg['hidden_f2'],
            f2_layers=cfg['layers_f2'],
            time_feat_dim=time_feat_dim,
            xy_harmonics=xy_harmonics,
            xy_include_input=xy_include_input,
        ).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        with torch.no_grad():
            latent = model.compute_latent_map(H, W, device=device)
            latent_np = latent.cpu().numpy()
    else:
        # 使用已导出的 latent_map.npy 和 f2_state_dict
        latent_np = np.load(args.latent_path)
        H, W, latent_dim = latent_np.shape
        if args.width is not None and args.height is not None and (args.width != W or args.height != H):
            print('[WARN] Provided width/height do not match latent_map.npy; will ignore width/height and use latent size.')
        W, H = latent_np.shape[1], latent_np.shape[0]

    # 构建或加载 F2
    if args.ckpt is not None:
        f2_state = None
        time_harmonics = cfg['time_harmonics']
        latent_dim = cfg['latent_dim']
        time_feat_dim = 2 * time_harmonics
        f2 = model.f2.to(device)
    else:
        f2_state = torch.load(args.f2_path, map_location=device)
        time_harmonics = args.time_harmonics if args.time_harmonics is not None else 2
        time_feat_dim = 2 * time_harmonics
        from src.models.compressor import F2Net
        f2 = F2Net(in_dim=latent_dim + time_feat_dim).to(device)
        f2.load_state_dict(f2_state)
        f2.eval()

    # 规范时间到 [0,24] 并归一化为 [0,1]
    t = float(args.time)
    t = max(0.0, min(24.0, t))
    t_norm = t / 24.0

    # 准备输入并批量渲染（矢量化以加速）
    latent = torch.from_numpy(latent_np).to(device)
    latent = latent.view(-1, latent.shape[-1])  # [H*W, latent_dim]

    # 时间特征为每个像素相同，但为向量化方便创建与 latent 对齐的张量
    t_feat = encode_time(torch.full((latent.shape[0],), t_norm, device=device), K=time_harmonics)  # [N, 2K]
    zt = torch.cat([latent, t_feat], dim=-1)

    with torch.no_grad():
        rgb = f2(zt)
    rgb = rgb.view(H, W, 3).clamp(0, 1).cpu().numpy()
    img = (rgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
