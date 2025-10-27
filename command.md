# 常用命令（统一使用 scripts 目录下的脚本）

以下命令均以 Windows cmd 为例，入口脚本统一位于 `scripts/` 目录，避免与根目录同名脚本混淆。

## 小批量训练测试（DeltaField 残差模式）
```
python scripts\train.py --data_dir D:\Project\AIRender\data\tod_output ^
	--out_dir D:\Project\AIRender\runs\delta_small ^
	--epochs 20 --batch_size 8192 --hidden 64 --layers 6 ^
	--time_harmonics 4 --xy_harmonics 4 --xy_include_input ^
	--lr 5e-4 --num_workers 1 --samples_per_epoch 200000 ^
	--residual_mode True --baseline_time 12
```

## 完整测试
```
python -m scripts.train --data_dir /home/lichengkai/AIRender/data/tod_output \
	--out_dir /home/lichengkai/AIRender/runs/delta
```

## 使用 checkpoint 来推理渲染
```
python scripts\infer.py --ckpt D:\Project\AIRender\runs\delta_small\best.ckpt ^
	--baseline_path D:\Project\AIRender\data\tod_output\tod_12.png ^
	--time 18 --out D:\Project\AIRender\runs\recon_18_with_base.png
```

## 批量推理渲染 0-23 时
```
python -m scripts.batch_infer --ckpt /home/lichengkai/AIRender/runs/delta/best.ckpt \
	--infer_script /home/lichengkai/AIRender/scripts/infer.py --output_dir /home/lichengkai/AIRender/runs/batch_infer \
	--start_hour 0 --end_hour 23
```

## 进行网络结构实验
```
python -m experiments.run_search --gpus 4 --num_workers 1 ^
	--space experiments\space.json --data_dir D:\Project\AIRender\data\tod_output ^
	--out_base runs\experiments
```
