#!/user/bin/python
# -* - coding:UTF-8 -*-
# 准备添加的数据
modify_data = open('file1.txt', 'r+')
lin = modify_data.readlines()
for i in range(500):
    part_one = lin[7*30 * i:7*30 * (i + 1)]

    # 修改inp文件
    file = open('compress.inp', 'r')  # 初始文件
    lines = file.readlines()
    # lines += part_one
    # print(lines)

    start1 = lines.index("** MATERIALS\n") + 1
    end1 = lines.index('** ----------------------------------------------------------------\n')
    # print(lines[start1:end1])
    lines[start1:end1] = '**\n'
    # print(lines[start1:end1])
    # 上面个就是找到了需要替换的部分以及 确定下来了已有部分，现在做的就是重新写inp文件将这两个进行替换
    #
    my_inp = open('compress-' + str(i + 1) + '.inp', 'w')
    for line in lines:
        if '** ----------------------------------------------------------------' in line:
            # print(line[:-448])
            line_ = part_one
            # print(line_)
            for j in range(len(line_)):
                my_inp.write(line_[j])
        else:
            my_inp.write(line)
    file.close()
    my_inp.close()
print('done')
