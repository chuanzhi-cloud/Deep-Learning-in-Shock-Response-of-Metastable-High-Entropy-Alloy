import numpy as np
import imageio

# 处理500张图片
for m in range(1):
    input_image_file = f"picture_{m + 1}.tiff"  # 输入的图片文件
    output_file = f"predict_file_{m + 1}.txt"  # 输出的txt文件
    try:
        # 使用 imageio 打开图片
        image = imageio.imread(input_image_file)
        print(image)
        # 确保图像是 RGB 图像
        if image.ndim == 2:  # 如果是灰度图像，转换为 RGB
            image = np.stack([image] * 3, axis=-1)

        # 将图像数据转换为浮动数值类型（float32）
        image_data = image.astype(np.float32)
        print(image_data)
        # 确保图像大小为 32x32 (根据你的要求，图片必须为 32x32)
        if image_data.shape[0] != 32 or image_data.shape[1] != 32:
            raise ValueError(f"图片 {input_image_file} 的尺寸不为32x32，无法处理。")

        # 将数据写入 txt 文件
        np.savetxt(output_file, image_data.reshape(-1, 3), fmt="%.4f")  # 每行保存 RGB 的浮动数值

        # 打印每个图片转换成功的提示
        print(f"图片 {input_image_file} 的数据已成功保存到 {output_file}")
    except Exception as e:
        print(f"无法处理图片 {input_image_file}: {e}")
