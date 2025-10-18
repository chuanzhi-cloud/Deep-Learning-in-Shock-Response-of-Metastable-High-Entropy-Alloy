# 里面有一些可以进行修改的参数，输入文件为得到的myoutput_file_{m + 1}.txt
# 输出为对应的图片，可修改参数为17，21，24，

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
# 输入和输出文件路径
for m in range(1):
    input_file = f"out_modify_texture_{m + 1}.txt"
    output_image_file = f"picture_{m + 1}.tiff"

    # 从 txt 文件读取数据，不进行归一化，直接加载
    data = np.loadtxt(input_file).astype(np.float32)

    # 检查数据行数是否为 1024 (32x32)
    if data.shape[0] != 1024 or data.shape[1] != 3:
        raise ValueError("输入数据格式错误，预期为 1024 行，每行 4 个值 (RGB)。")

    # 按行优先方式重塑为 32x32 图像
    image_data = data.reshape((32, 32, 3))
    # print(image_data)
    imageio.imwrite(output_image_file, image_data)  # 支持浮动数据
