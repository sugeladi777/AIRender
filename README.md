# AIRender — GridMLP: 多分辨率特征网格 + 小型 MLP 学习 ΔRGB(x,y,t)

本项目基于多分辨率特征网格（Multi-Resolution Feature Grids）+ 小型 MLP（GridMLP）来学习时空函数 f(x,y,t) → ΔRGB，其中 ΔRGB 是相对于“基准图”（如 t=12）的残差。推理时，将预测残差加回基准图即可重建任意时间的光照贴图。

## 主要文件说明

- `scripts/train.py`：训练 GridMLP 模型（学习 ΔRGB），会保存 `baseline.png` 与 `best.ckpt`。
- `scripts/infer.py`：使用 GridMLP `best.ckpt` 渲染指定时间图像，可选加载 `baseline.png`。
- `experiments/run_search.py`：并行实验与评估（支持 PSNR/SSIM/LPIPS），已适配 GridMLP。
- `src/models/grid_mlp.py`：GridMLP 与多分辨率特征网格实现，含批量渲染接口。
- `src/trainers/trainer.py`：训练循环与检查点保存（包含关键配置以支持推理恢复）。
- `src/data/dataset.py`：赛事 Data_HPRC 数据集读取（`config.json` + `.bin_0`）。

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

使用 HPRC 数据集（比赛提供的数据）
-----------------------------

本仓库已内置对组委会提供的 `Data_HPRC` 数据格式的支持（`config.json` + 二进制 `.bin_0` 文件）。

将 `--data_dir` 替换为 `--hprc_dir` 即可从该数据集训练：

```cmd
python -m scripts.train ^
  --hprc_dir D:\Project\AIRender\Data_HPRC ^
  --hprc_index 0 ^
  --out_dir D:\Project\AIRender\runs\hprc_demo ^
  --epochs 50 --batch_size 8192 ^
  --time_harmonics 8 --grid_levels 16,32,64,128 --channels_per_level 16 ^
  --lr 1e-4 --num_workers 8 --residual_mode --baseline_time 12
```

说明：
- `--hprc_dir` 指向包含 `config.json` 与 `Data/` 的目录
- `--hprc_index` 选择 `config.json` 中的第几个 lightmap（可先查看文件了解分辨率与 level 信息）
- 其余参数与普通训练一致；残差模式会自动保存 `baseline.png`

运行实验（网格搜索）
--------------------

仓库包含 `experiments/run_search.py`：根据 `experiments/space.json` 并行运行训练并评估（每个实验目录会包含训练输出与日志）。

示例（Windows CMD）：

```cmd
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4

python -m experiments.run_search ^
  --space experiments\space.json ^
  --hprc_dir D:\Project\AIRender\Data_HPRC ^
  --hprc_index 0 ^
  --out_base runs\experiments ^
  --gpu_ids 0,1 --num_workers 2
```


评估指标与可选依赖
-------------------

- PSNR：始终计算（NumPy 实现）。
- SSIM：如安装 `scikit-image` 则计算，否则跳过。
- LPIPS：如安装 `lpips` 则计算，否则跳过。

备注：
- 推理脚本 `scripts/infer.py` 会优先从 checkpoint 同目录读取 `baseline.png`；若不存在且 checkpoint 的 `config` 包含 `hprc_dir/hprc_index`，脚本会自动从 HPRC 数据读取对应基准帧。
- 批量推理 `scripts/batch_infer.py` 支持两种模式：
  - 默认整点渲染（start_hour..end_hour）
  - 指定 `--hprc_dir --hprc_index` 时按数据中的所有 `time_keys` 渲染（含 5.9 与 18.1）
