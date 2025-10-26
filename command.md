# 保存一些常用命令

## 小批量训练测试（DeltaField 残差模式）
`python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\delta_small --epochs 20 --batch_size 8192 --hidden 64 --layers 6 --time_harmonics 4 --xy_harmonics 4 --xy_include_input --lr 5e-4 --num_workers 1 --samples_per_epoch 200000 --residual_mode True --baseline_time 12`

## 完整测试（针对 RTX4060 推荐）
`python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\delta_4060 --epochs 100 --batch_size 65536 --hidden 64 --layers 6 --time_harmonics 4 --xy_harmonics 4 --xy_include_input --lr 5e-4 --num_workers 1 --residual_mode --baseline_time 12`

## 使用checkpoint来推理渲染
`python infer.py --ckpt D:\Project\AIRender\runs\delta_small\best.ckpt --baseline_path D:\Project\AIRender\dataset\tod_output\tod_12.png --time 18 --out D:\Project\AIRender\out\recon_18_with_base.png`

## 进行网络结构实验
`python -m experiments.run_search --gpus 4 --num_workers 1 --space experiments\space.json --data_dir D:\Project\AIRender\dataset\tod_output --out_base runs\experiments`
