import os
import subprocess
import argparse
import json
import sys

parser = argparse.ArgumentParser(description='批量渲染 HPRC 时刻或整点 24 小时结果')
parser.add_argument('--infer_script', type=str, default='scripts/infer.py', help='infer.py脚本路径')
parser.add_argument('--ckpt', type=str, default='runs/tod_cpu/best.ckpt', help='模型权重路径')
parser.add_argument('--output_dir', type=str, default='runs/batch_infer', help='输出图片文件夹')
parser.add_argument('--start_hour', type=int, default=0, help='起始小时')
parser.add_argument('--end_hour', type=int, default=23, help='结束小时')
parser.add_argument('--hprc_dir', type=str, default=None, help='如提供，则根据 HPRC 配置渲染所有 time_keys（含 5.9 与 18.1）')
parser.add_argument('--hprc_index', type=int, default=0, help='HPRC lightmap 索引（与 --hprc_dir 一起使用）')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def _render_one(time_hour: float, out_path: str):
    cmd = [
        sys.executable, args.infer_script,
        '--ckpt', args.ckpt,
        '--time', str(time_hour),
        '--out', out_path
    ]
    print(f'渲染 t={time_hour:.2f}: {out_path}')
    env = os.environ.copy()
    root = os.getcwd()
    old_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = root + (os.pathsep + old_pp if old_pp else '')
    subprocess.run(cmd, check=True, env=env)

if args.hprc_dir:
    # 根据 HPRC 配置渲染所有时刻
    cfg_path = os.path.join(args.hprc_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'config.json not found under {args.hprc_dir}')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    lst = cfg.get('lightmap_list', [])
    if not (0 <= args.hprc_index < len(lst)):
        raise IndexError(f'hprc_index {args.hprc_index} out of range [0, {len(lst)-1}]')
    item = lst[args.hprc_index]
    time_keys = sorted([int(k) for k in item['lightmaps'].keys()])
    for tk in time_keys:
        hour = tk / 100.0
        out = os.path.join(args.output_dir, f'render_{hour:05.2f}.png'.replace('.', '_'))
        _render_one(hour, out)
else:
    # 默认渲染整点 [start_hour, end_hour]
    for hour in range(args.start_hour, args.end_hour + 1):
        output_path = os.path.join(args.output_dir, f'render_{hour:02d}.png')
        _render_one(float(hour), output_path)

print('全部渲染完成，结果保存在', args.output_dir)
