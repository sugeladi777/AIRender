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

```
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

如需安装匹配 CUDA 的 PyTorch，请参考 https://pytorch.org/ 获取对应命令。

## 数据准备

将 24 张同尺寸的光照贴图放在一个文件夹（如 `data/daylight`），支持 `.png/.jpg/.jpeg`。文件名按字典序排序后应对应从 `t=0..23`。
```

## 训练

```
python train.py --data_dir data\\daylight --out_dir runs\\exp1 \
  --epochs 200 --batch_size 65536 --latent_dim 16 --hidden_f1 32 --hidden_f2 64 \
  --time_harmonics 2 --lr 1e-3
```

常用参数：
- `--latent_dim`：潜变量维度（默认32），越大重建更准但“中间变量贴图”更大。
- `--hidden_f1/hidden_f2`：F1/F2 隐藏层宽度（SIREN），越大会更准但实时成本更高。
- `--time_harmonics`：时间Fourier频次K（每个k产生sin/cos两维）。
- `--samples_per_epoch`：每个epoch随机采样的样本数（默认按图像总像素×24，可能很大，建议设置为几百万）。

训练结束会在 `out_dir` 下保存：
- `last.ckpt`/`best.ckpt`
- `latent_map.npy`：形状 `(H, W, latent_dim)` 的潜变量贴图。
- `f2_state_dict.pt` 与 `f2_ts.pt`（TorchScript）

## 导出（如需单独运行）
```
python export_latent.py --ckpt runs\\exp1\\best.ckpt --out_dir exports\\exp1
```

## 推理重建
```
python infer.py --latent_path exports\\exp1\\latent_map.npy \
  --f2_path exports\\exp1\\f2_state_dict.pt --time 12 \
  --width 256 --height 256 --out out\\recon_12.png
```

也可直接使用 `--ckpt` 一步渲染（内部先算latent）：
```
python infer.py --ckpt runs\\exp1\\best.ckpt --time 18 --out out\\recon_18.png
```

## 备注
- 所有坐标 `(x,y)` 归一化到 `[-1,1]`，时间 `t` 归一化到 `[0,1]` 并使用 sin/cos 编码。
- 输出采用 Sigmoid 到 `[0,1]`，损失为 MSE。
- 模型较小以利于实时：默认 F1 两层、F2 三层，可按需求调整。
