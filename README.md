# AIRender — 基于 SIREN 的 24 小时光照贴图压缩与重建

本仓库实现一种两阶段神经表示（coordinate-based / SIREN）：

- F1：把像素坐标 (x,y) 映射为每像素的潜变量向量 z（latent map）；
- F2：把 z 与时间特征 t 拼接后输出 RGB，用以重建任意时刻的光照贴图（24 帧/日）。

训练后可导出一张 `latent_map.npy`（H×W×latent_dim）和小型的 F2 模型（`f2_state_dict.pt` / `f2_ts.pt`），用于轻量部署。

## 主要文件说明

- `train.py`：训练脚本，支持 `--compute_latent` 在训练结束后导出 `latent_map.npy` / `f2_state_dict.pt` / `f2_ts.pt`。
- `export_latent.py`：从 checkpoint 单独导出 latent 与 F2（若训练时未导出）。
- `infer.py`：使用 checkpoint 或导出的 latent+F2 进行渲染。
- `experiments/run_search.py`：简单的实验网格搜索与评估（已支持按 PSNR/SSIM/LPIPS 排序结果）。
- `src/models/compressor.py`：F1/F2/组合模型与 `compute_latent_map`。
- `src/data/lightmap_dataset.py`：加载 24 帧数据并提供训练样本接口。

## 环境与依赖

我已把主要依赖整合到 `environment.yml`（conda）和 `pip_requirements.txt`（用于 PyTorch wheel 指定）。推荐在 Windows 上使用 conda 创建环境：
# AIRender — 基于 SIREN 的 24 小时光照贴图压缩与重建

简明说明
------------

本仓库实现一种两阶段神经表示（coordinate-based / SIREN）：

- F1：把像素坐标 (x,y) 映射为每像素的潜变量向量 z（latent map）。
- F2：把 z 与时间特征 t 拼接后输出 RGB，用以重建任意时刻的光照贴图（24 帧/日）。

训练后可导出一张 `latent_map.npy`（H×W×latent_dim）和小型的 F2 模型（`f2_state_dict.pt` / `f2_ts.pt`），用于轻量部署。

主要文件
-----------

- `train.py`：训练脚本，支持 `--compute_latent` 在训练结束后导出 `latent_map.npy` / `f2_state_dict.pt` / `f2_ts.pt`。
- `export_latent.py`：从 checkpoint 单独导出 latent 与 F2（若训练时未导出）。
- `infer.py`：使用 checkpoint 或导出的 latent+F2 进行渲染。
- `experiments/run_search.py`：简单的实验网格搜索与评估（会生成 `results.csv` 与 `results_sorted.csv`，包含 PSNR/SSIM/LPIPS）。
- `src/models/compressor.py`：F1/F2/组合模型与 `compute_latent_map`。
- `src/data/lightmap_dataset.py`：加载 24 帧数据并提供训练样本接口。

环境与依赖（建议）
------------------

主要依赖已整理在 `environment.yml`（conda）与 `pip_requirements.txt`（PyTorch wheel 指定）。推荐：

```cmd
conda env create -f environment.yml
conda activate AIRender
```

可选（用于完整评估指标）：

```cmd
pip install scikit-image lpips
```

- `scikit-image`：用于计算 SSIM（缺失时会跳过 SSIM）。
- `lpips`：用于计算 LPIPS（缺失时会跳过 LPIPS）。

快速开始（示例命令，Windows cmd.exe）
---------------------------------

准备数据：把 24 张同尺寸图像放在一个文件夹（按字典序对应 t=0..23），例如：

```
D:\Project\AIRender\dataset\tod_output
```

CPU 小规模调试：

```cmd
python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_cpu \
  --epochs 5 --batch_size 4096 --latent_dim 16 --hidden_f1 16 --hidden_f2 32 \
  --time_harmonics 2 --lr 1e-3 --num_workers 0 --samples_per_epoch 200000 --compute_latent
```

GPU 示例（显存充足时）：

```cmd
python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_gpu \
  --epochs 200 --batch_size 65536 --latent_dim 32 --hidden_f1 32 --hidden_f2 64 \
  --time_harmonics 4 --lr 5e-4 --num_workers 8 --compute_latent
```

从 checkpoint 导出 latent（若训练时未导出）：

```cmd
python export_latent.py --ckpt D:\Project\AIRender\runs\tod_cpu\best.ckpt --out_dir D:\Project\AIRender\exports\tod_cpu
```

推理（使用 checkpoint）：

```cmd
python infer.py --ckpt D:\Project\AIRender\runs\tod_cpu\best.ckpt --time 12 --out D:\Project\AIRender\out\recon_12.png
```

推理（使用导出 latent + F2）：

```cmd
python infer.py --latent_path D:\Project\AIRender\exports\tod_cpu\latent_map.npy --f2_path D:\Project\AIRender\exports\tod_cpu\f2_state_dict.pt --time 18 --out D:\Project\AIRender\out\recon_18.png
```

运行实验（网格搜索）
--------------------

仓库包含 `experiments/run_search.py`：根据 `experiments/space.json` 顺序逐项运行训练并评估（每个实验目录会包含训练输出及导出文件）。

推荐以模块方式运行（防止 `ModuleNotFoundError: No module named 'src'`）：

```cmd
python -m experiments.run_search --space experiments/space.json --data_dir D:\Project\AIRender\dataset\tod_output --out_base runs/experiments
```

要点：

- `run_search` 会强制在调用 `train.py` 时添加 `--compute_latent`，以便导出 `latent_map.npy` 与 `f2_ts.pt` 供评估使用；
- 汇总文件：`runs/experiments/results.csv`（包含 `psnr_mean, ssim_mean, lpips_mean` 字段）与 `runs/experiments/results_sorted.csv`（按 PSNR 降序、SSIM 降序、LPIPS 升序排序）。


评估指标与可选依赖
-------------------

- PSNR：始终计算（NumPy 实现）。
- SSIM：如安装 `scikit-image` 则计算，否则跳过。
- LPIPS：如安装 `lpips` 则计算，否则跳过。
