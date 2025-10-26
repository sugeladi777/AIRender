import os
import subprocess

import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='批量渲染24小时结果')
parser.add_argument('--infer_script', type=str, default='infer.py', help='infer.py脚本路径')
parser.add_argument('--ckpt', type=str, default='runs/tod_cpu/best.ckpt', help='模型权重路径')
parser.add_argument('--output_dir', type=str, default='batch_infer_output', help='输出图片文件夹')
parser.add_argument('--start_hour', type=int, default=0, help='起始小时')
parser.add_argument('--end_hour', type=int, default=23, help='结束小时')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

for hour in range(args.start_hour, args.end_hour + 1):
    output_path = os.path.join(args.output_dir, f'render_{hour:02d}.png')
    cmd = [
        'python', args.infer_script,
        '--ckpt', args.ckpt,
        '--time', str(hour),
        '--out', output_path
    ]
    print(f'正在渲染第 {hour} 小时: {output_path}')
    subprocess.run(cmd, check=True)

print('全部渲染完成，结果保存在', args.output_dir)

print('全部渲染完成，结果保存在', args.output_dir)