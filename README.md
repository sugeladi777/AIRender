# AIRender — 直接学习时空场 ΔRGB(x,y,t)（SIREN）

本项目采用单网络（SIREN MLP）直接学习函数 f(x,y,t) → ΔRGB，其中 ΔRGB 是相对某一“基准图”（例如 t=12） 的残差。推理时，将预测的残差加回基准图即可得到任意时间的重建结果。

## 主要文件说明

- `scripts/train.py`：训练单一 SIREN 模型 DeltaField（直接学习 ΔRGB），并保存 `baseline.png` 与 `best.ckpt`。
- `scripts/infer.py`：加载 `best.ckpt` 和 `baseline.png`，渲染指定时间图像。
- `experiments/run_search.py`：并行实验与评估（支持 PSNR/SSIM/LPIPS），已适配 DeltaField。
- `src/models/delta_field.py`：DeltaField 模型定义，内部使用 SIREN（`src/models/siren.py`）。
- `src/data/lightmap_dataset.py`：加载序列数据并在残差模式下返回 `rgb - baseline` 作为监督信号。

## 环境与依赖

主要依赖在 `pip_requirements.txt`，建议：

```cmd
pip install -r pip_requirements.txt
```

可选（用于完整评估指标）：

```cmd
pip install scikit-image lpips
```

- `scikit-image`：用于计算 SSIM（缺失时会跳过 SSIM）。
- `lpips`：用于计算 LPIPS（缺失时会跳过 LPIPS）。

## 快速开始（Windows cmd.exe）

准备数据：把 24 张同尺寸图像放在一个文件夹（按字典序对应 t=0..23），例如：

```
D:\Project\AIRender\dataset\tod_output
```

训练（默认残差模式，以 t=12 为基准）：

```cmd
python scripts\train.py --data_dir D:\Project\AIRender\data\tod_output --out_dir D:\Project\AIRender\runs\delta_t12 ^
  --epochs 100 --batch_size 65536 --hidden 64 --layers 6 ^
  --time_harmonics 4 --xy_harmonics 4 --xy_include_input ^
  --num_workers 1 --residual_mode --baseline_time 12
```

推理：

```cmd
python scripts\infer.py --ckpt D:\Project\AIRender\runs\delta_t12\best.ckpt --time 18 --out D:\Project\AIRender\runs\recon_18.png
```

运行实验（网格搜索）
--------------------

仓库包含 `experiments/run_search.py`：根据 `experiments/space.json` 并行运行训练并评估（每个实验目录会包含训练输出与日志）。

示例：

```cmd
set OMP_NUM_THREADS=7
set MKL_NUM_THREADS=7
set OPENBLAS_NUM_THREADS=7

python -m experiments.run_search --space experiments/space.json --data_dir D:\Project\AIRender\dataset\tod_output --out_base runs/experiments --gpu_ids 0,1,2,3 --num_workers 1
```


评估指标与可选依赖
-------------------

- PSNR：始终计算（NumPy 实现）。
- SSIM：如安装 `scikit-image` 则计算，否则跳过。
- LPIPS：如安装 `lpips` 则计算，否则跳过。
