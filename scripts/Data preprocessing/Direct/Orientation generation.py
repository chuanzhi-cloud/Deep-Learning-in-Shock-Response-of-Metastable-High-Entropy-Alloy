from math import pi
import random

grain_size = 30
sample_size = 200
file1 = open("file1.txt", "w")
file2 = open("file2.txt", "w")
for j in range(sample_size):

    # D= 300
    A = round(random.uniform(0.3, 2 * pi - 0.3), 2)
    B = round(random.uniform(0.3, pi - 0.3), 2)
    C = round(random.uniform(0.3, 2 * pi - 0.3), 2)
    D = 77
    for i in range(grain_size):
        disturbance_A = round(random.uniform(-0.1, 0.1), 2)  # 生成一个微小扰动，范围为-0.05到+0.05
        disturbance_B = round(random.uniform(-0.1, 0.1), 2)  # 生成一个微小扰动，范围为-0.05到+0.05
        disturbance_C = round(random.uniform(-0.1, 0.1), 2)  # 生成一个微小扰动，范围为-0.05到+0.05
        disturbance_D = 0  # 如果需要对 D 也加扰动，可以使用一个不同范围
        A = A + disturbance_A
        B = B + disturbance_B
        C = C + disturbance_C
        D = D + disturbance_D
        file1.write('*Material, name=MAT' + str(i + 1))
        file1.write('\n')
        file1.write('*Density')
        file1.write('\n')
        file1.write(' 7.655e-09,')
        file1.write('\n')
        file1.write('*Depvar')
        file1.write('\n')
        file1.write('    400,')
        file1.write('\n')
        file1.write('*User Material, constants=5')
        file1.write('\n')
        file1.write(f"{round(A, 2)}, {round(B, 2)}, {round(C, 2)}, 0, {round(D, 2)}")
        file1.write('\n')
        file2.write(f"{round(A, 2)}, {round(B, 2)}, {round(C, 2)}, 0, {round(D, 2)}")
        file2.write('\n')
file1.close()
file2.close()
