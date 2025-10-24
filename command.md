# 保存一些常用命令

## 小批量训练测试
`python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_cpu --epochs 50 --batch_size 8192 --latent_dim 64 --hidden_f1 64 --layers_f1 6 --hidden_f2 64 --layers_f2 6 --time_harmonics 4 --xy_harmonics 4 --xy_include_input --lr 5e-4 --num_workers 6 --samples_per_epoch 200000 --compute_latent`

## 完整测试（针对 RTX4060 推荐）
`python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_4060 --xy_include_input --num_workers 8 --compute_latent`

## 导出

## 使用checkpoint来推理渲染
`python infer.py --ckpt D:\Project\AIRender\runs\tod_4060\best.ckpt --time 12 --out  D:\Project\AIRender\out\recon_12.png`

## 使用导出的中间表示来推理渲染
`python infer.py --latent_path D:\Project\AIRender\runs\tod_cpu\latent_map.npy --f2_path D:\Project\AIRender\runs\tod_cpu\f2_state_dict.pt --time 18 --out D:\Project\AIRender\out\recon_18.png`
