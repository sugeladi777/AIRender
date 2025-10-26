from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch

from src.models.delta_field import DeltaField, DeltaFieldConfig
from src.utils.encoding import encode_time


def parse_args():
    """解析推理参数：支持从 ckpt 或 latent+f2 加载并渲染指定时间的图像。"""
    p = argparse.ArgumentParser(description='使用 DeltaField checkpoint 在给定时间重建图像')
    p.add_argument('--ckpt', type=str, required=True, help='训练产生的 .ckpt 文件路径（必需）')

    p.add_argument('--time', type=float, required=True, help='时间，单位小时，范围 [0,24]')
    p.add_argument('--out', type=str, required=True, help='输出图像路径（PNG）')
    p.add_argument('--baseline_path', type=str, default=None, help='残差模型时可选的 baseline.png 路径（默认从 ckpt 同目录读取）')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 校验输入
    if args.ckpt is None and (args.latent_path is None or args.f2_path is None):
        raise ValueError('Provide either --ckpt or both --latent_path and --f2_path')

    if args.ckpt is not None:
        # 使用 DeltaField 单模型推理（残差 + baseline）
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ckpt['config']
        H, W = ckpt['image_size']
        df_cfg = DeltaFieldConfig(
            time_harmonics=int(cfg.get('time_harmonics', 4)),
            xy_harmonics=int(cfg.get('xy_harmonics', 4)),
            xy_include_input=bool(cfg.get('xy_include_input', True)),
            hidden=int(cfg.get('hidden', 64) if 'hidden' in cfg else cfg.get('hidden_f2', 64)),
            layers=int(cfg.get('layers', 6) if 'layers' in cfg else cfg.get('layers_f2', 6)),
        )
        model = DeltaField(df_cfg).to(device)
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
                    print('[WARN] residual_mode=True 但未找到 baseline.png，将直接输出残差结果。')
    else:
        raise ValueError('DeltaField 模式仅支持 --ckpt 推理。')

    # 规范时间到 [0,24] 并归一化为 [0,1]
    t = float(args.time)
    t = max(0.0, min(24.0, t))
    t_norm = t / 24.0

    # 准备输入并批量渲染（矢量化以加速）
    with torch.no_grad():
        delta = DeltaField.render_image(model, H, W, t_norm, device=device)  # [H,W,3]
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
