# 项目目录结构说明

本项目采用「包源码在 src/，脚本在 scripts/，实验在 experiments/，文档在 docs/，输出在 runs/」的布局，避免脚本与包代码混淆，便于后续扩展与部署。

## 顶层目录
- src/                源码包（保持模块导入路径为 `src.*`，兼容现有代码）
  - data/            数据集与数据加载相关
  - models/          模型定义（SIREN、DeltaField 等）
  - utils/           工具方法（编码、通用工具）
  - trainers/        训练器（预留，当前为空，可后续扩展）
- scripts/            命令行脚本（训练、推理、批量推理）
  - train.py         训练入口（推荐使用）
  - infer.py         单次推理入口
  - batch_infer.py   批量推理入口（支持 0-23 小时）
- experiments/        实验自动化（搜索空间与批量运行器）
  - space.json       超参数搜索空间
  - run_search.py    批量试验脚本
- docs/               文档
  - STRUCTURE.md     本说明
  - 改进方向.md      未来模型/训练改进点记录
- data/               数据（示例/占位）
- runs/               训练与推理产物输出目录（检查点、图像）
- out/                旧版输出目录（若无使用可删除；推荐统一使用 runs/）
- pip_requirements.txt Python 依赖（可使用 `pip install -r pip_requirements.txt` 安装）
- environment.yml     Conda 环境（可选）
- README.md           简要说明
- command.md          常用命令（已统一指向 scripts/ 下脚本）

## 使用约定
- 统一从 `scripts/` 调用入口脚本：
  - 训练：`python scripts/train.py ...`
  - 推理：`python scripts/infer.py ...`
  - 批量：`python scripts/batch_infer.py ...`
- 源码内部以 `from src.xxx import yyy` 的形式导入，保持兼容。
- 产物统一输出到 `runs/`，避免多个输出目录混乱；`out/` 为旧目录，可迁移或删除。
