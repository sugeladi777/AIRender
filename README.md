# AIRender — GridMLP: 多分辨率特征网格 + 小型 MLP 学习 ΔRGB(x,y,t)

本项目基于多分辨率特征网格（Multi-Resolution Feature Grids）+ 小型 MLP（GridMLP）来学习时空函数 f(x,y,t) → ΔRGB，其中 ΔRGB 是相对于“基准图”（如 t=12）的残差。推理时，将预测残差加回基准图即可重建任意时间的光照贴图。

## 主要文件说明

- `scripts/train.py`：训练 GridMLP 模型（学习 ΔRGB），会保存 `baseline.png` 与 `best.ckpt`。
- `scripts/infer.py`：使用 GridMLP `best.ckpt` 渲染指定时间图像，可选加载 `baseline.png`。
- `experiments/run_search.py`：并行实验与评估（支持 PSNR/SSIM/LPIPS），已适配 GridMLP。
- `src/models/grid_mlp.py`：GridMLP 与多分辨率特征网格实现，含批量渲染接口。
- `src/trainers/trainer.py`：训练循环与检查点保存（包含关键配置以支持推理恢复）。
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

## 快速开始（Linux / macOS / Windows）

准备数据：把 24 张同尺寸图像放在一个文件夹（按字典序对应 t=0..23），例如：

```
D:\Project\AIRender\dataset\tod_output
```

训练（默认残差模式，以 t=12 为基准）：

```bash
python -m scripts.train \
  --data_dir /home/lichengkai/AIRender/data/tod_output \
  --out_dir /home/lichengkai/AIRender/runs/gridmlp_t12 \
  --epochs 150 --batch_size 16384 --hidden 256 --layers 8 \
  --time_harmonics 8 --grid_levels 16,32,64,128 --channels_per_level 16 \
  --lr 1e-4 --num_workers 8 --residual_mode True --baseline_time 12
```

推理：

```bash
python -m scripts.infer \
  --ckpt /home/lichengkai/AIRender/runs/gridmlp_t12/best.ckpt \
  --time 18 --out /home/lichengkai/AIRender/runs/recon_18.png
```

运行实验（网格搜索）
--------------------

仓库包含 `experiments/run_search.py`：根据 `experiments/space.json` 并行运行训练并评估（每个实验目录会包含训练输出与日志）。

示例（Linux）：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python -m experiments.run_search \
  --space experiments/space.json \
  --data_dir /home/lichengkai/AIRender/data/tod_output \
  --out_base runs/experiments \
  --gpu_ids 0,1,2,3 --num_workers 2
```


评估指标与可选依赖
-------------------

- PSNR：始终计算（NumPy 实现）。
- SSIM：如安装 `scikit-image` 则计算，否则跳过。
- LPIPS：如安装 `lpips` 则计算，否则跳过。
