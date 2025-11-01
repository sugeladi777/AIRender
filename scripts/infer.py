from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch

from src.models.grid_mlp import GridMLP, GridMLPConfig
from src.data.dataset import load_hprc_lightmap_spec


def parse_args():
    """解析推理参数：支持从 ckpt 或 latent+f2 加载并渲染指定时间的图像。"""
    p = argparse.ArgumentParser(description='使用 GridMLP checkpoint 在给定时间重建图像')
    p.add_argument('--ckpt', type=str, required=True, help='训练产生的 .ckpt 文件路径（必需）')

    p.add_argument('--time', type=float, required=True, help='时间，单位小时，范围 [0,24]')
    p.add_argument('--out', type=str, required=True, help='输出图像路径（PNG）')
    p.add_argument('--baseline_path', type=str, default=None, help='残差模型时可选的 baseline.png 路径（默认从 ckpt 同目录读取）')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 校验输入
    if args.ckpt is None:
        raise ValueError('Provide --ckpt for GridMLP inference')

    # 使用 GridMLP 推理（残差 + baseline）
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    H, W = ckpt['image_size']
    gm_cfg = GridMLPConfig(
        grid_levels=cfg.get('grid_levels', '16,32,64,128'),
        channels_per_level=int(cfg.get('channels_per_level', 16)),
        time_harmonics=int(cfg.get('time_harmonics', 8)),
        xy_harmonics=int(cfg.get('xy_harmonics', 0)),
        include_xy_input=bool(cfg.get('include_xy_input', False)),
        mlp_hidden=int(cfg.get('mlp_hidden', 64)),
        mlp_layers=int(cfg.get('mlp_layers', 3)),
        residual_mode=bool(cfg.get('residual_mode', True)),
    )
    model = GridMLP(gm_cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # baseline
    residual_mode = bool(cfg.get('residual_mode', True))
    baseline_img = None
    if residual_mode:
        if args.baseline_path and len(args.baseline_path) > 0 and os.path.exists(args.baseline_path):
            baseline_img = np.array(Image.open(args.baseline_path).convert('RGB'), dtype=np.float32) / 255.0
        else:
            base_candidate = Path(args.ckpt).parent / 'baseline.png'
            if base_candidate.exists():
                baseline_img = np.array(Image.open(base_candidate).convert('RGB'), dtype=np.float32) / 255.0
            else:
                # 尝试从 ckpt 配置中的 HPRC 数据读取 baseline 帧
                try:
                    hdir = cfg.get('hprc_dir', None)
                    hidx = int(cfg.get('hprc_index', 0))
                    btime = int(cfg.get('baseline_time', 12))
                    if hdir is not None:
                        spec = load_hprc_lightmap_spec(hdir, hidx)
                        base_key = int(round(float(btime) * 100))
                        if base_key not in spec.time_keys:
                            base_key = min(spec.time_keys, key=lambda k: abs(k - base_key))
                        lm_path = os.path.join(hdir if os.path.isdir(os.path.join(hdir, 'Data')) is False else os.path.join(hdir, 'Data'), spec.lightmaps[base_key])
                        H, W = H, W
                        data = np.fromfile(lm_path, dtype=np.float32)
                        baseline_img = data.reshape(H, W, 3)
                    else:
                        print('[WARN] residual_mode=True 但未找到 baseline.png，也未在 ckpt.config 中找到 hprc_dir，直接输出残差结果。')
                except Exception as e:
                    print('[WARN] 无法从 HPRC 数据读取 baseline：', e)

    # 规范时间到 [0,24] 并归一化为 [0,1]
    t = float(args.time)
    t = max(0.0, min(24.0, t))
    t_norm = t / 24.0

    # 准备输入并批量渲染（矢量化以加速）
    with torch.no_grad():
        delta = GridMLP.render_image(model, H, W, t_norm, device=device)  # [H,W,3]
    rgb = delta.cpu().numpy()

    # 如果是残差模型，推理结果为残差，需要加回 baseline
    if args.ckpt is not None and residual_mode:
        if 'baseline_img' in locals() and baseline_img is not None:
            if baseline_img.shape[0] != H or baseline_img.shape[1] != W:
                print('[WARN] baseline 尺寸与输出不一致，将进行缩放。')
                from PIL import Image as _Image
                baseline_img = np.array(_Image.fromarray((baseline_img*255).astype(np.uint8)).resize((W, H), resample=_Image.BILINEAR), dtype=np.float32) / 255.0
            rgb = rgb + baseline_img
        else:
            print('[WARN] 未提供 baseline 图，直接输出残差。')

    rgb = np.clip(rgb, 0.0, 1.0)
    img = (rgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
