import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='批量渲染24小时结果')
parser.add_argument('--infer_script', type=str, default='scripts/infer.py', help='infer.py脚本路径')
parser.add_argument('--ckpt', type=str, default='runs/tod_cpu/best.ckpt', help='模型权重路径')
parser.add_argument('--output_dir', type=str, default='runs/batch_infer', help='输出图片文件夹')
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
    # Ensure project root is on PYTHONPATH so `from src...` imports work when running the script
    env = os.environ.copy()
    root = os.getcwd()
    old_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = root + (os.pathsep + old_pp if old_pp else '')
    subprocess.run(cmd, check=True, env=env)

print('全部渲染完成，结果保存在', args.output_dir)
