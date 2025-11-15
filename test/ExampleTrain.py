import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os
import OpenEXR
import json

from ExampleModel import ExampleModel
import Utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default='../Data/SimpleData')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # 创建保存参数和结果的文件夹
    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)
     
    # 读取数据集下的配置文件
    config_file = 'config.json'
    time_count = 24
    with open(os.path.join(args.dataset, config_file), 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 初始化整个数据集的指标list
    total_psnr = []
    total_ssim = []
    total_lpips = []

    # 分别训练每张lightmap
    for lightmap in config['lightmap_list']:
        print(f"training lightmap {lightmap['level']}_{lightmap['id']}")

        # 从配置文件中获取lightmap的id、lightmap路径、mask路径、分辨率
        id = lightmap['id']
        lightmap_names = lightmap['lightmaps']
        mask_names = lightmap['masks']
        resolution = lightmap['resolution']

        # 读取每张lightmap在不同时间的数据
        # 关于读取数据，你可以参照ReadData.py来获取更详细的信息
        lightmap_in_different_time = []
        for time_idx in range(time_count):
            lightmap_path = os.path.join(args.dataset, lightmap_names[str(time_idx)])
            lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
            lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3))
        lightmap_data = torch.from_numpy(np.concatenate(lightmap_in_different_time, axis=0)).to(torch.float32).to(device)

        # 读取mask数据
        mask_in_different_time = []
        for time_idx in range(time_count):
            mask_path = os.path.join(args.dataset, mask_names[str(time_idx)])
            mask_bin = np.fromfile(mask_path, dtype=np.int8)
            mask_in_different_time.append(mask_bin.reshape(resolution['height'], resolution['width']))
        mask_data = np.concatenate(mask_in_different_time, axis=0).reshape(time_count, resolution['height'], resolution['width'])

        # 初始化模型
        model = ExampleModel(input_dim=6, output_dim=3, hidden_dim=args.hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 生成归一化坐标，并组合时间信息
        xs, ys = np.meshgrid(np.arange(resolution['width']), np.arange(resolution['height']))
        coords = np.stack([ys / (resolution['height'] - 1), xs / (resolution['width'] - 1)], axis=-1).reshape(-1, 2)
        coords = torch.from_numpy(coords).to(torch.float32).to(device)
        total_coords = []
        for time_idx in range(time_count):
            alpha = torch.full((resolution['width'] * resolution['height'], 1), (time_idx / (time_count - 1))).to(device)
            coords_with_time = torch.cat([coords, alpha], dim=-1)
            total_coords.append(coords_with_time)
        total_coords = torch.cat(total_coords, dim=0)
        total_data = torch.cat([total_coords, lightmap_data], dim=-1)
        total_data =total_data[torch.randperm(total_data.shape[0])]
        
        # 训练循环
        batch_start = 0
        for it in range(args.iterations):
            batch_end = min(batch_start + args.batch_size, total_coords.shape[0])
            batch_data = total_data[batch_start:batch_end]

            pred = model(batch_data[:, :3])
            loss = criterion(pred, batch_data[:, 3:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start = batch_end
            if batch_start >= total_coords.shape[0]:
                batch_start = 0
                total_data = total_data[torch.randperm(total_data.shape[0])]

            if (it + 1) % 1000 == 0:
                print(f"iteration {it + 1} loss: {loss.item()}")

        # 保存实际模型参数为二进制文件
        # 文件大小就是模型实际的大小，也是计算压缩率的依据
        # 请不要进行任何压缩
        all_params = []
        for param in model.parameters():
            all_params.append(param.detach().cpu().numpy().flatten())
        params_array = np.concatenate(all_params)
        params_array.astype(np.float32).tofile(f"./Parameters/model_{lightmap['level']}_{id}_params.bin")

        # 测试阶段
        with torch.no_grad():
            # 完整重建整张lightmap
            model.eval()
            pred_list = []
            for i in range((total_coords.shape[0] + args.batch_size - 1) // args.batch_size):
                batch_start = i * args.batch_size
                batch_end = min(batch_start + args.batch_size, total_coords.shape[0])
                batch_data = total_coords[batch_start:batch_end]
                pred = model(batch_data[:, :3])
                pred_list.append(pred)
            pred = torch.cat(pred_list, dim=0)
            pred = pred.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)

            # 将lightmap数据reshape为[time_count, height, width, 3]方便计算指标
            lightmap_data = lightmap_data.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)
            
            # 初始化该lightmap的指标list
            psnr_list = []
            ssim_list = []
            lpips_list = []

            # 计算指标，每256*256的区域计算一次
            part_size = 256
            rows = (lightmap_data.shape[2] + part_size - 1) // part_size
            cols = (lightmap_data.shape[3] + part_size - 1) // part_size
            for time_idx in range(time_count):
                # 无效区域置0
                pred[time_idx, :, mask_data[time_idx] <= 0] = 0
                for i in range(rows):
                    for j in range(cols):
                        start_row = i * part_size
                        end_row = min((i + 1) * part_size, lightmap_data.shape[2])
                        start_col = j * part_size
                        end_col = min((j + 1) * part_size, lightmap_data.shape[3])

                        lightmap_part = lightmap_data[[time_idx], :, start_row:end_row, start_col:end_col]
                        lightmap_reconstruct_part = pred[[time_idx], :, start_row:end_row, start_col:end_col]
                        mask_part = mask_data[time_idx, start_row:end_row, start_col:end_col]
                        valid_mask = mask_part >= 127

                        # 可以忽略完全无效的区域
                        if (np.any(valid_mask) and lightmap_part.max() != 0):
                            psnr_list.append(Utils.cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part))
                            ssim_list.append(Utils.cal_ssim(lightmap_part, lightmap_reconstruct_part))
                            lpips_list.append(Utils.cal_lpips(lightmap_part, lightmap_reconstruct_part))

            # 将该lightmap的指标list添加到整个数据集的指标list中
            total_psnr.extend(psnr_list)
            total_ssim.extend(ssim_list)
            total_lpips.extend(lpips_list)
            
            # 打印该lightmap的指标
            print(f"metrics of lightmap {lightmap['level']}_{id}------------")
            print(f"PSNR: {np.mean(psnr_list)}")
            print(f"SSIM: {np.mean(ssim_list)}")
            print(f"LPIPS: {np.mean(lpips_list)}")
            print(f"Model Size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")
            print(f"Data Size: {lightmap_data.shape[0] * lightmap_data.shape[1] * lightmap_data.shape[2] * lightmap_data.shape[3] * 4 / 1024 / 1024:.2f} MB")
            print("-----------------------------------------")

            # 保存拟合结果为exr文件
            pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            for time_idx in range(time_count):
                path = f'./ResultImages/reconstructed_{id}_{time_idx + 1:02d}.00.exr'
                header = OpenEXR.Header(pred[time_idx].shape[1], pred[time_idx].shape[0])
                channels = ['R', 'G', 'B']
                exr = OpenEXR.OutputFile(path, header)
                exr.writePixels({
                    c: pred[time_idx][..., i].tobytes()
                    for i, c in enumerate(channels)
                })
                exr.close()

    # 打印整个数据集的指标
    print(f"metrics of total data set ---------------")        
    print(f"PSNR of all lightmaps: {np.mean(total_psnr)}")
    print(f"SSIM of all lightmaps: {np.mean(total_ssim)}")
    print(f"LPIPS of all lightmaps: {np.mean(total_lpips)}")
    print("-----------------------------------------")
    
if __name__ == "__main__":
    main()