import json
import os
import time
import numpy as np
import torch

import Interface
import Utils

if __name__ == '__main__': 
    # 所有大赛使用的数据集
    data_set_list = ['Data_HPRC']
    test_time = list(range(24)) + [5.9, 18.1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data_set in data_set_list:
        # 读取数据集下的配置文件
        dataset_path = f'../Data/{data_set}'
        print(f"dataset_path Set: {dataset_path}")
        config_file = 'config.json'
        with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        lightmap_count = data['lightmap_count']
        
        # 初始化整个数据集的指标list
        psnr_list = []
        ssim_list = []
        lpips_list = []
        time_list = []
        data_size = 0

        # 计算所有lightmap的指标
        for lightmap_config in data['lightmap_list']:
            print(f"Level Name: {lightmap_config['level']}, LightMap ID: {lightmap_config['id']}")
            # 从配置文件中获取lightmap的id、lightmap路径、mask路径、分辨率
            lightmap_names = lightmap_config['lightmaps']
            mask_names = lightmap_config['masks']
            resolution = lightmap_config['resolution']

            # 计算数据大小，3通道，24个时刻，每个通道4个字节
            data_size += resolution['height'] * resolution['width'] * 3 * 4 * 24

            # 获取接口对象
            your_interface = Interface.get(lightmap_config, device)
            
            # 分别重建每个时刻的lightmap
            for current_time in test_time:
                with torch.no_grad():
                    # 计算推理时间
                    torch.cuda.synchronize()
                    time_start = time.time()
                    your_interface.reconstruct(current_time)

                    # 获取重建结果
                    lightmap_reconstruct = your_interface.get_result()

                    torch.cuda.synchronize()
                    time_end = time.time()
                    time_list.append(time_end - time_start)

                    # 测试Random Access
                    random_coord = np.zeros((1, 3))
                    random_y = np.random.randint(0, resolution['height'])
                    random_x = np.random.randint(0, resolution['width'])
                    random_coord[0, 0] = random_y
                    random_coord[0, 1] = random_x
                    random_coord[0, 2] = current_time
                    random_lightmap_reconstruct = your_interface.random_test(random_coord)
                    if not torch.allclose(random_lightmap_reconstruct[0, :], lightmap_reconstruct[0, :, random_y, random_x], atol=1e-2):
                        print(f"Random Access Test Failed at {current_time} {random_y} {random_x} !!!")
                        exit()
                    
                    # 读取原始lightmap数据计算指标
                    lightmap_path = os.path.join(dataset_path, "Data", lightmap_names[str(int(current_time * 100))])
                    mask_path = os.path.join(dataset_path, "Data", mask_names[str(int(current_time * 100))])
                                    
                    lightmap_data = np.fromfile(lightmap_path, dtype=np.float32)
                    mask_data = np.fromfile(mask_path, dtype=np.int8)
                    lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
                    mask = mask_data.reshape(resolution['height'], resolution['width'])

                    # 每256*256分辨率计算一次指标，最后取平均值
                    # 我们在计算指标的时候不会考虑无效的像素，所以你可以在你的训练中不考虑这些无效像素
                    lightmap_reconstruct[:, :, mask <= 0] = 0
                    lightmap = torch.from_numpy(lightmap).permute(2, 0, 1).unsqueeze(0).to(device)

                    part_size = 256
                    rows = (lightmap.shape[2] + part_size - 1) // part_size
                    cols = (lightmap.shape[3] + part_size - 1) // part_size
                    for i in range(rows):
                        for j in range(cols):
                            start_row = i * part_size
                            end_row = min((i + 1) * part_size, lightmap.shape[2])
                            start_col = j * part_size
                            end_col = min((j + 1) * part_size, lightmap.shape[3])

                            lightmap_part = lightmap[:, :, start_row:end_row, start_col:end_col]
                            lightmap_reconstruct_part = lightmap_reconstruct[:, :,start_row:end_row, start_col:end_col]
                            mask_part = mask[start_row:end_row, start_col:end_col]
                            valid_mask = mask_part >= 127

                            if (np.any(valid_mask) and lightmap_part.max() != 0):
                                psnr_list.append(Utils.cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part))
                                ssim_list.append(Utils.cal_ssim(lightmap_part, lightmap_reconstruct_part))
                                lpips_list.append(Utils.cal_lpips(lightmap_part, lightmap_reconstruct_part))

        # 获取模型大小，用于计算压缩率
        model_size = Utils.get_folder_size('./Parameters')

        print(f"Data Set: {data_set}")
        print(f"PSNR: {np.mean(psnr_list)}")
        print(f"SSIM: {np.mean(ssim_list)}")
        print(f"LPIPS: {np.mean(lpips_list)}")
        print(f"Time: {np.sum(time_list)}")
        print(f"Model Size: {model_size} Bytes")
        print(f"Data Size: {data_size} Bytes")
        print(f"Compression Ratio: {model_size / data_size:.4f}")

