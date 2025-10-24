# 保存一些常用命令

## 小批量训练测试
`python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_cpu --epochs 5 --batch_size 4096 --latent_dim 16 --hidden_f1 16 --hidden_f2 32 --time_harmonics 2 --lr 1e-3 --num_workers 0 --samples_per_epoch 200000 --compute_latent`

## 完整测试
`python train.py --data_dir D:\Project\AIRender\dataset\tod_output --out_dir D:\Project\AIRender\runs\tod_4060 --epochs 2 --batch_size 16384 --latent_dim 32 --hidden_f1 32 --hidden_f2 64 --time_harmonics 2 --lr 1e-3 --num_workers 8 --compute_latent --amp`

## 导出

## 使用checkpoint来推理渲染
`python infer.py --ckpt D:\Project\AIRender\runs\tod_cpu\best.ckpt --time 12 --out D:\Project\AIRender\out\recon_12.png`

## 使用导出的中间表示来推理渲染
`python infer.py --latent_path D:\Project\AIRender\runs\tod_cpu\latent_map.npy --f2_path D:\Project\AIRender\runs\tod_cpu\f2_state_dict.pt --time 18 --out D:\Project\AIRender\out\recon_18.png`