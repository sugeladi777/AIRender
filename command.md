# 常用命令（统一使用 scripts 目录下的脚本）

以下命令提供 Linux 与 Windows 示例，入口脚本统一位于 `scripts/` 目录，避免与根目录同名脚本混淆。

## 小批量训练测试（GridMLP 残差模式）
Linux:
```bash
python -m scripts.train \
	--data_dir /home/lichengkai/AIRender/data/tod_output \
	--out_dir /home/lichengkai/AIRender/runs/gridmlp_small \
	--epochs 20 --batch_size 8192 --hidden 64 --layers 3 \
	--time_harmonics 4 --grid_levels 16,32,64,128 --channels_per_level 16 \
	--lr 5e-4 --num_workers 2 --samples_per_epoch 200000 \
	--residual_mode True --baseline_time 12
```

## 完整测试
```bash
python -m scripts.train --data_dir /home/lichengkai/AIRender/data/tod_output \
  --out_dir /home/lichengkai/AIRender/runs/gridmlp_full
```

## 使用 checkpoint 来推理渲染
```bash
python -m scripts.infer --ckpt /home/lichengkai/AIRender/runs/gridmlp_small/best.ckpt \
  --baseline_path /home/lichengkai/AIRender/data/tod_output/tod_12.png \
  --time 5 --out /home/lichengkai/AIRender/runs/recon_5_with_base.png
```

## 批量推理渲染 0-23 时
```bash
python -m scripts.batch_infer --ckpt /home/lichengkai/AIRender/runs/gridmlp_full/best.ckpt \
  --infer_script /home/lichengkai/AIRender/scripts/infer.py --output_dir /home/lichengkai/AIRender/runs/batch_infer \
  --start_hour 0 --end_hour 23
```

## 进行网络结构实验
Linux:
```bash
python -m experiments.run_search --gpus 4 --num_workers 2 \
  --space experiments/space.json --data_dir /home/lichengkai/AIRender/data/tod_output \
  --out_base runs/experiments
```
