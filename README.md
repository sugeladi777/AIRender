# 基于 SIREN 的24小时光照贴图压缩

本项目实现一种两阶段神经压缩：
- F1: 将像素坐标 `(x,y)` 映射为中间变量（潜变量 `z`）。
- F2: 接收 `(z, t)`，输出像素 `rgb`，其中 `t` 是时间（一天24帧）。

训练完成后，可导出一张“潜变量贴图（latent map）”和 F2 的模型参数，实现用一张中间变量贴图 + 一个小模型重建全天光照。

## 目录结构

- `src/models/siren.py`：SIREN 层与 MLP 实现。
- `src/models/compressor.py`：F1/F2/组合模型定义。
- `src/data/lightmap_dataset.py`：24帧光照贴图数据集加载与采样。
- `src/utils/time_encoding.py`：时间编码（Fourier sin/cos）。
- `train.py`：训练脚本，支持导出潜变量贴图与权重。
- `export_latent.py`：从检查点导出潜变量贴图与 F2 模型（含 TorchScript）。
- `infer.py`：用导出的潜变量贴图 + F2 重建任意时刻图像。

## 环境准备（Windows, cmd）

建议使用 Python 3.10+/CUDA 环境。
# AIRender — 基于 SIREN 的 24 小时光照贴图压缩与重建

一个轻量级的两阶段神经表示（neural representation）项目：

- F1（坐标到潜变量）：把像素坐标 (x,y) 映射到每像素的潜变量向量 z；
- F2（潜变量+时间到颜色）：把 z 与时间特征拼接后输出 RGB，能在任意时刻重建光照贴图。

目标是使用一张 `latent_map.npy`（形状为 H×W×latent_dim）加上一个小型的 F2 模型，来表达并实时重建全天 24 小时的光照变化，从而大幅减少存储与部署成本。

## 项目结构（主要文件）

- `train.py`：训练脚本（支持在训练结束后导出 latent 与 F2）。
- `export_latent.py`：从 checkpoint 导出 `latent_map.npy`、`f2_state_dict.pt` 和 `f2_ts.pt`（TorchScript）。
- `infer.py`：使用 `latent_map.npy` + `f2_state_dict.pt` 或直接使用 `.ckpt` 渲染任意时刻的图像。
- `src/models/siren.py`：SIREN（sin 激活）相关实现。
- `src/models/compressor.py`：F1/F2/组合模型与 `compute_latent_map` 实现。
- `src/data/lightmap_dataset.py`：加载 24 帧光照图并返回训练样本。
- `src/utils/time_encoding.py`：时间 Fourier 编码（sin/cos）。

## 从零到完成：一步步操作（Windows, cmd.exe）

### 1) 准备环境

推荐使用 Python 3.11。先创建并激活虚拟环境：

```cmd
conda create -n AIRender python=3.11
pip install -r requirements.txt
```

### 2) 准备数据

把 24 张同尺寸的光照贴图放到一个文件夹，例如：

```
D:\Project\AIRender\dataset\tod_output
```

文件支持 `.png/.jpg/.jpeg`；按字典序排序后对应 t=0..23（如果不是 24 张，代码会给出警告但仍按排序进行索引）。

### 3) 训练（示例）

快速 CPU 调试（小规模）：
```cmd
python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_cpu \
  --epochs 5 --batch_size 4096 --latent_dim 16 --hidden_f1 16 --hidden_f2 32 \
  --time_harmonics 2 --lr 1e-3 --num_workers 0 --samples_per_epoch 200000 --compute_latent
```

常规 GPU 训练（如果有 CUDA 且显存充足）：
```cmd
python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_gpu \
  --epochs 200 --batch_size 65536 --latent_dim 32 --hidden_f1 32 --hidden_f2 64 \
  --time_harmonics 2 --lr 1e-3 --num_workers 0 --compute_latent
```

说明：
- `--compute_latent`：训练结束后自动计算并保存 `latent_map.npy`、`f2_state_dict.pt` 和 `f2_ts.pt` 到 `--out_dir`。
- 如遇 OOM（显存不足），请将 `--batch_size`、`--latent_dim` 或隐藏单元数调小，或降低 `--samples_per_epoch`。

训练输出（`--out_dir`）通常包含：
- `best.ckpt`, `last.ckpt`（检查点）
- `latent_map.npy`（若使用 `--compute_latent` 或通过 `export_latent.py` 导出）
- `f2_state_dict.pt`, `f2_ts.pt`

### 4) 从 checkpoint 导出 latent_map（可选）

如果你没有在训练时开启 `--compute_latent`，可以用 `export_latent.py` 单独导出：

```cmd
python export_latent.py --ckpt D:\Project\AIRender\runs\tod_cpu\best.ckpt --out_dir D:\Project\AIRender\exports\tod_cpu
```

导出后 `out_dir` 下会包含：
- `latent_map.npy`（H×W×latent_dim）
- `f2_state_dict.pt`（PyTorch state_dict）
- `f2_ts.pt`（TorchScript，可直接部署）

### 5) 推理 / 重建（示例）

直接用 checkpoint（一键计算 latent 并渲染）：

```cmd
python infer.py --ckpt D:\Project\AIRender\runs\tod_cpu\best.ckpt --time 12 --out D:\Project\AIRender\out\recon_12.png
```

使用已导出的 `latent_map.npy` + `f2_state_dict.pt`（更快、适合部署）：

```cmd
python infer.py --latent_path D:\Project\AIRender\exports\tod_cpu\latent_map.npy --f2_path D:\Project\AIRender\exports\tod_cpu\f2_state_dict.pt --time 18 --out D:\Project\AIRender\out\recon_18.png
```
