import json
import os
import numpy as np
import OpenEXR

# 读取lightmap文件样例
dataset_path = '../Data/SimpleData'
config_file = 'config.json'

# 读取数据集下的config文件，获取数据集配置参数
with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取lightmap数量
lightmap_count = data['lightmap_count']
print(f"Lightmap count: {lightmap_count}")
time = 9

# 获取每张lightmap的配置参数
for lightmap in data['lightmap_list']:
    id = lightmap['id']
    lightmap_names = lightmap['lightmaps']
    masks_names = lightmap['masks']
    resolution = lightmap['resolution']

    # 获取path信息
    lightmap_path = os.path.join(dataset_path, lightmap_names[str(time)])
    mask_path = os.path.join(dataset_path, masks_names[str(time)])

    # lightmap数据类型为float32
    lightmap_data = np.fromfile(lightmap_path, dtype=np.float32)
    # mask数据类型为int8
    mask_data = np.fromfile(mask_path, dtype=np.int8)

    # lightmap每个像素有R G B三通道
    lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
    # mask每个像素有1通道
    mask = mask_data.reshape(resolution['height'], resolution['width'])

    # mask数据为-1时，表示该数据为无效数据，为127时，表示该数据为有效数据
    # 获取有效lightmap数据可以这样做：
    valid_lightmap = lightmap[mask >= 127]

    # 你可以将这些数据保存为exr文件以供查看：
    R = lightmap[:, :, 0].tobytes()
    G = lightmap[:, :, 1].tobytes()
    B = lightmap[:, :, 2].tobytes()
    exr_file = OpenEXR.OutputFile(f'lightmap_{id}_{time}.exr', 
                                  OpenEXR.Header(resolution['width'], resolution['height']))
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()

