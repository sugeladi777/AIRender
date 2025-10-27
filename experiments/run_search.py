#!/usr/bin/env python3
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
from src.models.grid_mlp import GridMLP, GridMLPConfig


def build_cmd(config, data_dir, out_dir_base, num_workers=None):
    out_name = config.get('name') or f"exp_{int(time.time())}"
    run_out = os.path.join(out_dir_base, out_name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_out, exist_ok=True)

    # 使用模块方式调用，确保路径稳定：python -m scripts.train
    cmd = [sys.executable, '-m', 'scripts.train']
    # required args
    cmd += ['--data_dir', data_dir]
    cmd += ['--out_dir', run_out]

    # map rest of config
    # 只传递 train.py 支持的参数
    valid_keys = {
        'epochs', 'batch_size', 'hidden', 'layers', 'time_harmonics',
        'grid_levels', 'channels_per_level',
        'lr', 'weight_decay', 'scheduler_patience', 'scheduler_factor', 'min_lr', 'clip_grad',
        'samples_per_epoch', 'num_workers', 'seed', 'residual_mode', 'baseline_time', 'save_every'
    }
    # 需要单独处理的布尔 flag（store_true / store_false 风格）
    # 对于这些 key：当值为 True 时仅添加不带值的 `--key`；当值为 False 时添加 `--no_key`
    flag_keys = {'residual_mode'}
    for k, v in config.items():
        if v is None or k == 'name' or k not in valid_keys:
            continue
        if k in flag_keys:
            # 对于 flag：当 True 时添加 `--residual_mode`，当 False 时添加 `--no_residual_mode`
            if bool(v):
                cmd.append(f'--{k}')
            else:
                cmd.append(f'--no_{k}')
            continue
        # 其它参数以 --key value 形式传递；布尔值显式为 True/False
        cmd.append(f'--{k}')
        if isinstance(v, bool):
            cmd.append('True' if v else 'False')
        else:
            cmd.append(str(v))
    if '--num_workers' not in cmd and num_workers is not None:
        cmd.append('--num_workers')
        cmd.append(str(num_workers))
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
        cfg = best.get('config', {})
        ds = LightMapTimeDataset(
            image_dir=data_dir,
            time_harmonics=cfg.get('time_harmonics', 2),
            sample_mode='all',
            residual_mode=False,  # 评估时先加载原始 GT 栈
        )
        stack = ds.stack  # [T,H,W,3]
    except Exception as e:
        print(f"[WARN] Failed to load dataset for evaluation: {e}")
        return None, None, None

    T, H, W, _ = stack.shape

    # Build GridMLP model from checkpoint
    cfg = best.get('config', {})
    is_grid = (best.get('model_type') == 'grid_mlp') or ('grid_levels' in cfg)
    model = None
    if is_grid:
        try:
            gm_cfg = GridMLPConfig(
                grid_levels=cfg.get('grid_levels', '16,32,64,128'),
                channels_per_level=int(cfg.get('channels_per_level', 16)),
                time_harmonics=int(cfg.get('time_harmonics', 8)),
                mlp_hidden=int(cfg.get('mlp_hidden', cfg.get('hidden', 64))),
                mlp_layers=int(cfg.get('mlp_layers', cfg.get('layers', 3))),
                residual_mode=bool(cfg.get('residual_mode', True)),
            )
            model = GridMLP(gm_cfg)
            model.load_state_dict(best['model'])
            model.eval()
        except Exception as e:
            print(f"[WARN] Could not build GridMLP from checkpoint: {e}")
            return None, None, None

    # 如果是残差模型，准备 baseline
    residual_mode = bool(best.get('config', {}).get('residual_mode', False))
    baseline_img = None
    if residual_mode:
        baseline_idx = int(best.get('config', {}).get('baseline_time', 12))
        base_file = os.path.join(run_out, 'baseline.png')
        if os.path.exists(base_file):
            try:
                from PIL import Image as _Image
                baseline_img = np.array(_Image.open(base_file).convert('RGB'), dtype=np.float32) / 255.0
            except Exception:
                baseline_img = None
        if baseline_img is None:
            # fallback: 从数据集中取对应帧
            if 0 <= baseline_idx < T:
                baseline_img = stack[baseline_idx]

    # evaluate across frames
    psnrs = []
    ssims = []
    preds = []
    for t in range(T):
        t_norm = float(t / 24.0)
        try:
            if is_grid and model is not None:
                with torch.no_grad():
                    delta = GridMLP.render_image(model, H, W, t_norm, device=torch.device('cpu'))
                out_np = delta.numpy()
            else:
                print('[WARN] Unsupported checkpoint type for evaluation.')
                return None, None, None
        except Exception as e:
            print(f"[WARN] Failed to render frame {t}: {e}")
            return None, None, None

        gt = stack[t]
        # 如果是残差模型，推理输出需加回 baseline
        if residual_mode and baseline_img is not None:
            out_np = out_np + baseline_img
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
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to run in parallel (uses CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU ids to use (overrides --gpus)')
    parser.add_argument('--num_workers', type=int, default=None, help='Override DataLoader num_workers for all runs (overrides space.json)')
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

        # Prepare GPU list for parallel runs
        gpu_list = []
        if args.gpu_ids:
            try:
                gpu_list = [int(x) for x in args.gpu_ids.split(',') if x.strip() != '']
            except Exception:
                gpu_list = []
        else:
            if torch.cuda.is_available():
                avail = torch.cuda.device_count()
                take = min(args.gpus, avail) if args.gpus > 0 else 1
                gpu_list = list(range(take))
            else:
                gpu_list = []

        max_concurrent = len(gpu_list) if len(gpu_list) > 0 else 1
        print(f"Running experiments using up to {max_concurrent} concurrent GPU workers: {gpu_list if gpu_list else 'CPU only'}")

        running = []  # list of dicts: {proc, log_fh, run_out, start, idx, cfg, gpu}

        def spawn_experiment(cmd, run_out, gpu_assigned, idx, cfg):
            log_path = os.path.join(run_out, 'run.log')
            os.makedirs(run_out, exist_ok=True)
            env = os.environ.copy()
            # set CUDA_VISIBLE_DEVICES if GPU assigned
            if gpu_assigned is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_assigned)
            # 设置并行线程数，避免过度抢占；允许通过 cfg['omp_threads'] 覆盖
            omp_threads = str(cfg.get('omp_threads', 1))
            env.setdefault('OMP_NUM_THREADS', omp_threads)
            env.setdefault('MKL_NUM_THREADS', omp_threads)
            env.setdefault('OPENBLAS_NUM_THREADS', omp_threads)
            fh = open(log_path, 'w', encoding='utf-8')
            print('Spawning:', ' '.join(shlex.quote(x) for x in cmd), f'on GPU {gpu_assigned}')
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env)
            return {'proc': proc, 'log_fh': fh, 'run_out': run_out, 'start': time.time(), 'idx': idx, 'cfg': cfg, 'log_path': log_path, 'gpu': gpu_assigned}

        next_gpu_idx = 0
        for idx, cfg in enumerate(space):
            try:
                cmd, run_out = build_cmd(cfg, args.data_dir, args.out_base, num_workers=args.num_workers if hasattr(args, 'num_workers') else None)

                # wait for a free slot if needed
                while len(running) >= max_concurrent:
                    time.sleep(2)
                    # poll running procs
                    for r in running[:]:
                        ret = r['proc'].poll()
                        if ret is not None:
                            # finished
                            r['log_fh'].close()
                            duration = time.time() - r['start']
                            final_loss = extract_final_loss(r['log_path'])
                            rc = r['proc'].returncode
                            # evaluate
                            try:
                                psnr_mean, ssim_mean, lpips_mean = evaluate_run(r['run_out'], args.data_dir)
                            except Exception as e:
                                print('Evaluation failed:', e)
                                psnr_mean, ssim_mean, lpips_mean = None, None, None

                            writer.writerow({
                                'id': r['idx'],
                                'name': r['cfg'].get('name'),
                                'run_out': r['run_out'],
                                'returncode': rc,
                                'duration_s': f'{duration:.1f}',
                                'final_loss': final_loss if final_loss is not None else '',
                                'psnr_mean': f'{psnr_mean:.4f}' if psnr_mean is not None else '',
                                'ssim_mean': f'{ssim_mean:.4f}' if ssim_mean is not None else '',
                                'lpips_mean': f'{lpips_mean:.6f}' if lpips_mean is not None else '',
                                'config_json': json.dumps(r['cfg'], ensure_ascii=False),
                                'log_path': r['log_path'],
                            })
                            csvfile.flush()
                            running.remove(r)

                # assign GPU id (round-robin) or None
                gpu_assigned = None
                if len(gpu_list) > 0:
                    gpu_assigned = gpu_list[next_gpu_idx % len(gpu_list)]
                    next_gpu_idx += 1

                job = spawn_experiment(cmd, run_out, gpu_assigned, idx, cfg)
                running.append(job)

            except Exception as e:
                print('Experiment failed to start', e)
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

        # wait for remaining running jobs
        while len(running) > 0:
            time.sleep(2)
            for r in running[:]:
                ret = r['proc'].poll()
                if ret is not None:
                    r['log_fh'].close()
                    duration = time.time() - r['start']
                    final_loss = extract_final_loss(r['log_path'])
                    rc = r['proc'].returncode
                    try:
                        psnr_mean, ssim_mean, lpips_mean = evaluate_run(r['run_out'], args.data_dir)
                    except Exception as e:
                        print('Evaluation failed:', e)
                        psnr_mean, ssim_mean, lpips_mean = None, None, None

                    writer.writerow({
                        'id': r['idx'],
                        'name': r['cfg'].get('name'),
                        'run_out': r['run_out'],
                        'returncode': rc,
                        'duration_s': f'{duration:.1f}',
                        'final_loss': final_loss if final_loss is not None else '',
                        'psnr_mean': f'{psnr_mean:.4f}' if psnr_mean is not None else '',
                        'ssim_mean': f'{ssim_mean:.4f}' if ssim_mean is not None else '',
                        'lpips_mean': f'{lpips_mean:.6f}' if lpips_mean is not None else '',
                        'config_json': json.dumps(r['cfg'], ensure_ascii=False),
                        'log_path': r['log_path'],
                    })
                    csvfile.flush()
                    running.remove(r)

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
