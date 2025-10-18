import re
from math import *
# 定义晶粒个数
grain_num = 30
# 读取txt文件
file_path = 'element information.txt'  # 替换为你的文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
# 正则表达式提取elset编号和数字
pattern = r"\*Elset, elset=Set-Grain-(\d+)\s+([\d,\s]+)"
# 查找所有匹配
matches = re.findall(pattern, data)
# 将 elset 编号和数字列表存储为字典 {elset编号: 数字列表}
elset_ranges = {}

for match in matches:
    elset_number = int(match[0])  # 获取 elset 编号
    numbers = [int(num.strip()) for num in match[1].replace('\n', ',').split(',') if num.strip()]
    elset_ranges[elset_number] = numbers


# 创建一个函数来判断数字属于哪个elset
def find_elset(number):
    for elset_number, numbers in elset_ranges.items():
        if number in numbers:
            return elset_number
    return None  # 如果没有找到所属的elset

# 由于单元编号是从最后一行开始的，所以先创建一个列表使得单元编号顺序正确
numbers = []
# 定义开始数字和步长
start = 1024
step = 32

# 按照要求写入递增数字
for _ in range(32):  # 总共1024个数字，每次递减32个数字
    end = start - step + 1
    numbers.extend(range(end, start + 1))  # 向列表中添加从end到start的递增数字
    start = end - 1  # 更新start为当前段的结束值

# 如果这里只需要前三个通道的话就修改33，34，48行的三部分值，只进行前三个数据的写入
coefficients = [2 * pi, pi, 2 * pi]  # 弧度转角度
coefficients2 = [255, 255, 255]  # 彩色图片的归一化参数

for m in range(1):
    output_file = f"predict_texture_{m + 1}.txt"   # 这里是我想要的文件，应该是30行
    with open(output_file, 'w+') as f:
        f.writelines(['\n'] * grain_num)
        # 将列表中的数据写入工作表的第一行
        row_infor = []
        new_data_file = f"predict_file_{m + 1}.txt"  # 替换为您的新数据文件路径
        with open(new_data_file, 'r', encoding='utf-8') as file:
            new_data = [list(map(float, line.strip().split(' '))) for line in file.readlines()]
        for index, i in enumerate(numbers):
            elset = find_elset(i)
            print(f"Number {i} belongs to Elset-{elset}")
            if elset is not None:
                row_infor = [new_data[index][j] for j in [0, 1, 2]]
                modified_row_infor = [row_infor[k] * coefficients[k] for k in range(len(coefficients))]
                print(modified_row_infor)
                # modified_row_infor = [modified_row_infor[k] * coefficients2[k] for k in range(len(coefficients2))]
            # f.write(" ".join(f"{value:.4f}" for value in modified_row_infor) + "\n")
            print("-"*100)
            f.seek(0, 0)  # 先返回文件开头
            lines = f.readlines()  # 读取所有行
            formatted_data = ",".join(f"{value:.2f}" for value in modified_row_infor)+"\n"
            lines[elset-1] = formatted_data
            # 重新将修改后的内容写回
            f.seek(0, 0)  # 文件指针回到开头
            f.writelines(lines)