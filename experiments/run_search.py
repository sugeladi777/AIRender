#!/usr/bin/env python3
"""
Simple experiment runner for AIRender `train.py`.

Usage:
  python experiments/run_search.py --space experiments/space.json --data_dir <path/to/data> --out_base runs/experiments

What it does:
 - Reads a JSON list of experiment configurations (see experiments/space.json)
 - For each config it creates a unique output directory and runs `python train.py` with those args
 - Captures stdout/stderr to a log file and extracts the last reported `loss=` value as `final_loss`
 - Writes summary `results.csv` in the out_base directory

Notes:
 - This runner is intentionally simple and sequential (no parallelism). You can run multiple instances in parallel
   if you have multiple GPUs and adjust the commands accordingly.
 - The script expects to be run from repository root. Adjust paths if needed.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
import re
import math
from typing import Optional

import numpy as np
import torch

from src.data.lightmap_dataset import LightMapTimeDataset
from src.utils.encoding import encode_time


def build_cmd(config, data_dir, out_dir_base):
    out_name = config.get('name') or f"exp_{int(time.time())}"
    run_out = os.path.join(out_dir_base, out_name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_out, exist_ok=True)

    cmd = [sys.executable, 'train.py']
    # required args
    cmd += ['--data_dir', data_dir]
    cmd += ['--out_dir', run_out]

    # map rest of config
    for k, v in config.items():
        if v is None:
            continue
        if k == 'name':
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k}')
        else:
            cmd.append(f'--{k}')
            cmd.append(str(v))

    # Ensure we export latent_map and F2 (needed for evaluation)
    if '--compute_latent' not in cmd:
        cmd.append('--compute_latent')

    return cmd, run_out


def extract_final_loss(log_path):
    # scan the log for occurrences like "loss=0.00538" and return the last one
    loss_re = re.compile(r'loss=([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)')
    last = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                for m in loss_re.finditer(line):
                    last = float(m.group(1))
    except FileNotFoundError:
        return None
    return last


def _psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0:
        return float('inf')
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_psnr(img_gt: np.ndarray, img_pred: np.ndarray) -> float:
    # imgs are H,W,3 in [0,1]
    mse = float(np.mean((img_gt - img_pred) ** 2))
    return _psnr_from_mse(mse, data_range=1.0)


def compute_ssim(img_gt: np.ndarray, img_pred: np.ndarray) -> Optional[float]:
    # try skimage first
    try:
        from skimage.metrics import structural_similarity as ssim
        val = ssim(img_gt, img_pred, multichannel=True, data_range=1.0)
        return float(val)
    except Exception:
        # skimage not available or failed
        return None


def compute_lpips_stack(imgs_gt: np.ndarray, imgs_pred: np.ndarray) -> Optional[float]:
    """
    Compute average LPIPS across frames using the 'lpips' package (PyTorch).
    imgs_*: array [T,H,W,3] in [0,1]
    Returns mean LPIPS or None if lpips not available.
    """
    try:
        import lpips
    except Exception:
        return None

    # create model
    try:
        loss_fn = lpips.LPIPS(net='alex')
    except Exception:
        loss_fn = lpips.LPIPS(net='vgg')

    loss_fn.eval()
    vals = []
    with torch.no_grad():
        for i in range(imgs_gt.shape[0]):
            a = imgs_pred[i].astype(np.float32)
            b = imgs_gt[i].astype(np.float32)
            # to torch: [1,3,H,W], map [0,1] -> [-1,1]
            ta = torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0) * 2.0 - 1.0
            tb = torch.from_numpy(b.transpose(2,0,1)).unsqueeze(0) * 2.0 - 1.0
            try:
                v = loss_fn(ta, tb)
                vals.append(float(v.mean().cpu().numpy()))
            except Exception:
                return None
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def evaluate_run(run_out: str, data_dir: str):
    """Evaluate trained run by loading latent_map and F2 and computing PSNR/SSIM/LPIPS over all frames.

    Returns tuple (psnr_mean, ssim_mean_or_None, lpips_mean_or_None)
    """
    # load best checkpoint
    best_path = os.path.join(run_out, 'best.ckpt')
    if not os.path.exists(best_path):
        best_path = os.path.join(run_out, 'last.ckpt')
        if not os.path.exists(best_path):
            print(f"[WARN] No checkpoint found in {run_out}")
            return None, None, None

    try:
        best = torch.load(best_path, map_location='cpu')
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint {best_path}: {e}")
        return None, None, None

    # load dataset stack (ground truth)
    try:
        ds = LightMapTimeDataset(image_dir=data_dir, time_harmonics=best.get('config', {}).get('time_harmonics', 2), sample_mode='all')
        stack = ds.stack  # [T,H,W,3]
    except Exception as e:
        print(f"[WARN] Failed to load dataset for evaluation: {e}")
        return None, None, None

    T, H, W, _ = stack.shape

    # load latent map
    latent_path = os.path.join(run_out, 'latent_map.npy')
    if os.path.exists(latent_path):
        latent = np.load(latent_path)  # [H, W, latent_dim]
    else:
        # try to compute latent by loading model (fallback)
        print(f"[WARN] latent_map.npy not found in {run_out}; trying to compute from checkpoint (may be slow)")
        try:
            from src.models.compressor import F1F2Model
            cfg = best.get('config', {})
            latent_dim = int(cfg.get('latent_dim', 64))
            time_feat_dim = 2 * int(cfg.get('time_harmonics', 2))
            model = F1F2Model(
                latent_dim=latent_dim,
                f1_hidden=int(cfg.get('hidden_f1', 64)),
                f1_layers=int(cfg.get('layers_f1', 4)),
                f2_hidden=int(cfg.get('hidden_f2', 64)),
                f2_layers=int(cfg.get('layers_f2', 4)),
                time_feat_dim=time_feat_dim,
                xy_harmonics=int(cfg.get('xy_harmonics', 0)),
                xy_include_input=bool(cfg.get('xy_include_input', False)),
            )
            model.load_state_dict(best['model'])
            model.eval()
            with torch.no_grad():
                latent_t = model.compute_latent_map(H, W, device=torch.device('cpu'))
            latent = latent_t.numpy()
        except Exception as e:
            print(f"[WARN] Could not compute latent_map: {e}")
            return None, None, None

    # prefer traced F2 torchscript
    f2_ts_path = os.path.join(run_out, 'f2_ts.pt')
    f2_fn = None
    use_torchscript = False
    if os.path.exists(f2_ts_path):
        try:
            f2_ts = torch.jit.load(f2_ts_path, map_location='cpu')
            f2_fn = lambda x: f2_ts(x)
            use_torchscript = True
        except Exception:
            f2_fn = None

    # fallback: try to build F2 from checkpoint
    if f2_fn is None:
        try:
            from src.models.compressor import F1F2Model
            cfg = best.get('config', {})
            latent_dim = int(cfg.get('latent_dim', 64))
            time_feat_dim = 2 * int(cfg.get('time_harmonics', 2))
            model = F1F2Model(
                latent_dim=latent_dim,
                f1_hidden=int(cfg.get('hidden_f1', 64)),
                f1_layers=int(cfg.get('layers_f1', 4)),
                f2_hidden=int(cfg.get('hidden_f2', 64)),
                f2_layers=int(cfg.get('layers_f2', 4)),
                time_feat_dim=time_feat_dim,
                xy_harmonics=int(cfg.get('xy_harmonics', 0)),
                xy_include_input=bool(cfg.get('xy_include_input', False)),
            )
            model.load_state_dict(best['model'])
            model.eval()
            def _f2_call(x: torch.Tensor) -> torch.Tensor:
                # x: [N, latent_dim+time_feat_dim]
                with torch.no_grad():
                    return model.f2(x)
            f2_fn = _f2_call
        except Exception as e:
            print(f"[WARN] Could not build F2 from checkpoint: {e}")
            return None, None, None

    # evaluate across frames
    psnrs = []
    ssims = []
    preds = []
    for t in range(T):
        t_norm = float(t / 24.0)
        t_feat = encode_time(t_norm, K=best.get('config', {}).get('time_harmonics', 2))
        # latent: [H,W,latent_dim]
        z = latent.reshape(-1, latent.shape[-1])  # [H*W, latent_dim]
        # tile time
        tf = np.array(t_feat, dtype=np.float32)
        tf_tile = np.broadcast_to(tf, (z.shape[0], tf.shape[0]))  # [H*W, time_feat_dim]
        inp = np.concatenate([z, tf_tile], axis=-1)
        # to torch and run f2
        try:
            tin = torch.from_numpy(inp).to(dtype=torch.float32)
            tout = f2_fn(tin)
            out_np = tout.cpu().numpy().reshape(H, W, 3)
        except Exception as e:
            print(f"[WARN] Failed to run F2 for frame {t}: {e}")
            return None, None, None

        gt = stack[t]
        # clamp
        out_np = np.clip(out_np, 0.0, 1.0)
        preds.append(out_np)
        psnrs.append(compute_psnr(gt, out_np))
        ssim_v = compute_ssim(gt, out_np)
        if ssim_v is not None:
            ssims.append(ssim_v)

    preds = np.stack(preds, axis=0)  # [T,H,W,3]
    # LPIPS across stack (if available)
    lpips_mean = compute_lpips_stack(stack, preds)

    psnr_mean = float(np.mean([p for p in psnrs if np.isfinite(p)])) if len(psnrs) > 0 else None
    ssim_mean = float(np.mean(ssims)) if len(ssims) > 0 else None

    return psnr_mean, ssim_mean, lpips_mean


def run_experiment(cmd, run_out):
    log_path = os.path.join(run_out, 'run.log')
    start = time.time()
    print('Running:', ' '.join(shlex.quote(x) for x in cmd))
    with open(log_path, 'w', encoding='utf-8') as lg:
        proc = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=os.getcwd())
        proc.wait()
        rc = proc.returncode
    duration = time.time() - start
    final_loss = extract_final_loss(log_path)
    return rc, duration, final_loss, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', type=str, default='experiments/space.json', help='JSON file with list of configs')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--out_base', type=str, default='runs/experiments', help='Base output dir for experiments')
    parser.add_argument('--space_id', type=int, default=None, help='If set, only run that index in the space (0-based)')
    args = parser.parse_args()

    os.makedirs(args.out_base, exist_ok=True)

    with open(args.space, 'r', encoding='utf-8') as f:
        space = json.load(f)

    if args.space_id is not None:
        space = [space[args.space_id]]

    results_csv = os.path.join(args.out_base, 'results.csv')
    fieldnames = ['id', 'name', 'run_out', 'returncode', 'duration_s', 'final_loss', 'psnr_mean', 'ssim_mean', 'lpips_mean', 'config_json', 'log_path']

    # append mode so you can resume
    with open(results_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.path.getsize(results_csv) == 0:
            writer.writeheader()

        for idx, cfg in enumerate(space):
            try:
                cmd, run_out = build_cmd(cfg, args.data_dir, args.out_base)
                rc, duration, final_loss, log_path = run_experiment(cmd, run_out)
                # evaluate metrics (PSNR/SSIM/LPIPS)
                try:
                    psnr_mean, ssim_mean, lpips_mean = evaluate_run(run_out, args.data_dir)
                except Exception as e:
                    print('Evaluation failed:', e)
                    psnr_mean, ssim_mean, lpips_mean = None, None, None

                writer.writerow({
                    'id': idx,
                    'name': cfg.get('name'),
                    'run_out': run_out,
                    'returncode': rc,
                    'duration_s': f'{duration:.1f}',
                    'final_loss': final_loss if final_loss is not None else '',
                    'psnr_mean': f'{psnr_mean:.4f}' if psnr_mean is not None else '',
                    'ssim_mean': f'{ssim_mean:.4f}' if ssim_mean is not None else '',
                    'lpips_mean': f'{lpips_mean:.6f}' if lpips_mean is not None else '',
                    'config_json': json.dumps(cfg, ensure_ascii=False),
                    'log_path': log_path,
                })
                csvfile.flush()
            except Exception as e:
                print('Experiment failed', e)
                writer.writerow({
                    'id': idx,
                    'name': cfg.get('name'),
                    'run_out': '',
                    'returncode': -1,
                    'duration_s': '',
                    'final_loss': '',
                    'config_json': json.dumps(cfg, ensure_ascii=False),
                    'log_path': '',
                })

    print('Finished experiments. Results saved to', results_csv)
    # produce a sorted CSV according to PSNR(desc), SSIM(desc), LPIPS(asc)
    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
        def _key(row):
            # convert fields, empty -> very small / large to push to end
            try:
                psnr = float(row.get('psnr_mean') or -1e9)
            except Exception:
                psnr = -1e9
            try:
                ssim = float(row.get('ssim_mean') or -1e9)
            except Exception:
                ssim = -1e9
            try:
                lp = float(row.get('lpips_mean') or 1e9)
            except Exception:
                lp = 1e9
            # sort by PSNR desc, SSIM desc, LPIPS asc
            return (-psnr, -ssim, lp)

        sorted_rows = sorted(reader, key=_key)
        sorted_csv = os.path.join(args.out_base, 'results_sorted.csv')
        with open(sorted_csv, 'w', newline='', encoding='utf-8') as f:
            if len(sorted_rows) > 0:
                keys = sorted_rows[0].keys()
            else:
                keys = fieldnames
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in sorted_rows:
                w.writerow(r)
        print('Sorted results saved to', sorted_csv)
    except Exception as e:
        print('Could not produce sorted results:', e)


if __name__ == '__main__':
    main()
