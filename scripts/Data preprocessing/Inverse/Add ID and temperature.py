# 读取原始文件并处理数据
input_file = 'predict_texture_1.txt'  # 输入文件
output_file = 'predict_textureall_1.txt'  # 输出文件

with open(input_file, 'r', encoding='utf-8') as file:
    # 读取文件中的所有行
    lines = file.readlines()

# 打开输出文件进行写操作
with open(output_file, 'w', encoding='utf-8') as file:
    for line in lines:
        # 去掉每行末尾的换行符并按逗号分割
        parts = line.strip().split(',')

        # 添加固定的两个元素 0 和 300
        parts.append('0')
        parts.append('300')

        # 将修改后的内容重新组合成字符串并写入输出文件
        file.write(','.join(parts) + '\n')

print("文件处理完成，结果已写入", output_file)
