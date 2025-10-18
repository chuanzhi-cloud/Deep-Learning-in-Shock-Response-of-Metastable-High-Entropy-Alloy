import os
import datetime
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import csv
import tifffile

out_dir = os.path.join('./', datetime.datetime.now().strftime('%Y%m%d-%H%M') + "out")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


# 定义加载图片的函数
def load_tiff_images(input_folder):
    image_list = []  # 用于存储图片的列表

    # 获取input文件夹下所有tiff文件，按自然顺序排序
    tiff_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith('.tiff')],
        key=lambda x: [int(i) if i.isdigit() else i for i in re.split('([0-9]+)', x)]
    )
    # 加载每个tiff图片并将其加入列表
    for tiff_file in tiff_files:
        tiff_path = os.path.join(input_folder, tiff_file)
        # 使用tifffile库读取图片，假设每张tiff图片有四个通道
        img = tifffile.imread(tiff_path)
        # 确保图片是四个通道
        if img.ndim == 3 and img.shape[2] == 4:
            img_normalized = (img - 0.5) * 2  # 假设像素值在[0, 1]范围
            image_list.append(img_normalized)
        else:
            print(f"Warning: {tiff_file} is not a 4-channel image.")

    return np.array(image_list)


# 调用函数进行数据加载
input_folder = 'input'  # 文件夹路径
images = load_tiff_images(input_folder)
# # 打印已加载图片的数量
# print(f"Loaded {len(images)} images with 4 channels each.")
#
# # 打印第一张图片的名字及其数据
# first_image_data = images[0]
# print(f"First image data:\n{first_image_data}")


# 读取CSV文件
csv_file = './output/应力应变曲线.csv'  # 替换为你的CSV文件路径
stress = []
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    # 遍历CSV文件中的每一行
    for row in reader:
        # 提取每行的后60个数据
        last_60_data = [round(float(value) / 1900, 4) for value in row[:]]
        # 按照隔一个取一个的规则
        filtered_data = last_60_data[::1]
        # 将选取的数据添加到列表中
        stress.append(filtered_data)
# # 打印前几行提取的数据
# for row in stress[:5]:
#     print(row)


# 读取CSV文件
csv_file = './output2/相变体积分数.csv'  # 替换为你的CSV文件路径
phase = []
with open(csv_file, 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    # 遍历CSV文件中的每一行
    for row in reader:
        # 提取每行的后60个数据
        last_60_data = [round(float(value), 4) for value in row[:]]
        # 按照隔一个取一个的规则
        filtered_data = last_60_data[::1]
        # 将选取的数据添加到列表中
        phase.append(filtered_data)
# # 打印前几行提取的数据
# for row in phase[:5]:
#     print(row)

# # print(np.array(phase.shape, np.array(phase).shape)
response = np.concatenate((np.array(stress), np.array(phase)), axis=1)
# 设置随机种子，确保所有随机操作可复现
seed = 97
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)
# 加载数据集
# 将 NumPy 数组转换为 PyTorch 张量
input_tensor = torch.tensor(images, dtype=torch.float32)  # 转换为 float32 类型
input_tensor = input_tensor.permute(0, 3, 1, 2)  # 交换轴 1 和 3
output_tensor = torch.tensor(response, dtype=torch.float32)  # 转换为 float32 类型
# 使用 TensorDataset 创建数据集
dataset = TensorDataset(input_tensor, output_tensor)
# 计算训练集、验证集和测试集的样本数
total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% 训练集
val_size = int(0.2 * total_size)  # 20% 验证集
test_size = total_size - train_size - val_size  # 10% 测试集
# 使用 random_split 划分数据集，同时传入固定的随机生成器
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)
# 创建 DataLoader 以便分批加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)


# 将划分好的数据再次进行保存，便于统计
import numpy as np
import pandas as pd
#
#
# 假设train_dataset, val_dataset, test_dataset已经定义好了
# 这些数据集的 output_tensor 部分就是你想保存的数据（应力和相变数据）

def save_dataset_to_csv(dataset, file_path):
    data_list = []

    for _, output in dataset:
        # 提取每个样本的数据并将其添加到列表中
        data_list.append(output.numpy())

    # 合并所有的数据，将其转换为 NumPy 数组，并按行（样本数）存储
    all_data = np.array(data_list)

    # 将 NumPy 数组转换为 DataFrame，并按行保存
    df = pd.DataFrame(all_data)
    df.to_csv(file_path, index=False, header=False)


# 文件路径
train_csv_path = './train_data.csv'
val_csv_path = './val_data.csv'
test_csv_path = './test_data.csv'

# 保存训练集、验证集、测试集的数据
save_dataset_to_csv(train_dataset, train_csv_path)
save_dataset_to_csv(val_dataset, val_csv_path)
save_dataset_to_csv(test_dataset, test_csv_path)
# # 保存相关的图片（这里是测试集）
import os
import tifffile

# 定义保存测试集图片的文件夹
save_folder = './testpicture'
os.makedirs(save_folder, exist_ok=True)

# 遍历测试集中的每个样本
# 注意：test_dataset中的每个样本是 (input_tensor, output_tensor)
for i, (img_tensor, _) in enumerate(test_dataset):
    # img_tensor 的形状为 (4, H, W)，需要转换回 (H, W, 4)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    # 构造文件名，例如 "test_1.tiff", "test_2.tiff", ...
    file_name = f"test_{i+1}.tiff"
    file_path = os.path.join(save_folder, file_name)
    # 保存图片，格式与读取时一致
    tifffile.imsave(file_path, img_np)
#
#
class CNN_DualBranch(nn.Module):
    def __init__(self):
        super(CNN_DualBranch, self).__init__()

        # 特征提取部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4))  # 输出特征尺寸为 (128,4,4)
        )

        # 全连接特征融合层
        self.fc_common = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        # 分支1
        self.branch1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 30),
        )

        # 分支2
        self.branch2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 30),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        common_features = self.fc_common(x)

        out1 = self.branch1(common_features)
        out2 = self.branch2(common_features)

        return out1, out2


# 创建模型实例
# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_DualBranch().to(device)


# 自定义损失函数
def weighted_mse_loss(prediction, target, weight_type='linear'):
    length = 30
    # 生成权重
    if weight_type == 'linear':
        # 使用线性加权：前面的元素权重较小，后面的元素权重大
        weights = torch.linspace(1.0, 5.0, steps=length).to(prediction.device)
    elif weight_type == 'exponential':
        # 使用指数加权：前面权重小，后面权重大
        weights = torch.exp(torch.linspace(0.0, 2.0, steps=length)).to(prediction.device)
    else:
        # 默认为1，即不使用权重
        weights = torch.ones(length).to(prediction.device)
    # 计算误差
    error = prediction - target
    # 计算加权的平方误差
    weighted_error = weights * error ** 2
    # 返回加权平均误差
    loss = weighted_error.mean()
    return loss


# 自定义损失函数
# 使用均方误差作为损失函数，Adam优化器
# criterion = nn.MSELoss()
criterion = weighted_mse_loss
# optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 定义学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
# 训练模型
num_epochs = 200
train_losses = []
val_losses = []
train_errors = []  # 用于存储训练集的误差
val_errors = []  # 用于存储验证集的误差
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_error = 0.0  # 训练误差

    # 训练集
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # 通过模型得到两个输出
        stress_output, phase_output = model(inputs)

        # 计算两个输出的损失
        loss_stress = criterion(stress_output, targets[:, :30])
        loss_phase = criterion(phase_output, targets[:, 30:])
        loss = loss_stress + loss_phase  # 总损失是两个损失之和

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算误差（例如，绝对误差）
        error_stress = torch.abs(stress_output - targets[:, :30]).mean()
        error_phase = torch.abs(phase_output - targets[:, 30:]).mean()
        running_error += (error_stress + error_phase).item()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_error = running_error / len(train_loader)  # 平均误差
    train_losses.append(avg_train_loss)
    train_errors.append(avg_train_error)

    # 验证集
    model.eval()
    val_loss = 0.0
    running_val_error = 0.0  # 验证误差
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            stress_output, phase_output = model(inputs)
            # 计算两个输出的损失
            loss_stress = criterion(stress_output, targets[:, :30])
            loss_phase = criterion(phase_output, targets[:, 30:])
            # 加入加权版本（注意权重要放到相同 device）
            # w = weight_vector.to(targets.device)  # 放到当前GPU或CPU上

            # 广播机制：每个 batch 都乘同样的 weight
            # loss_stress = ((stress_output - targets[:, :30]) ** 2 * w).mean() * 1.5
            # loss_phase = ((phase_output - targets[:, 30:]) ** 2 * w).mean()
            val_loss += loss_stress.item() + loss_phase.item()

            # 计算验证集的误差
            error_stress = torch.abs(stress_output - targets[:, :30]).mean()
            error_phase = torch.abs(phase_output - targets[:, 30:]).mean()
            running_val_error += (error_stress + error_phase).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_error = running_val_error / len(val_loader)  # 平均误差
    val_losses.append(avg_val_loss)
    val_errors.append(avg_val_error)
    # 根据验证损失调整学习率
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Training Loss: {avg_train_loss:.4f}, Training Error: {avg_train_error:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}, Validation Error: {avg_val_error:.4f}")

# 保存模型
model_name = 'cnn_regressor_model.pth'
model_path = os.path.join(out_dir, model_name)
torch.save(model.state_dict(), model_path)
# 保存训练历史到txt文件
textname = "train_history.txt"  # 输出每个样本的文件名
filepathtext = os.path.join(out_dir, textname)
with open(filepathtext, "w") as f:
    f.write("Epoch,Train Loss,Val Loss,Train Error,Val Error\n")  # 写入表头
    for epoch in range(num_epochs):
        f.write(
            f"{epoch + 1},{train_losses[epoch]:.4f},{val_losses[epoch]:.4f},{train_errors[epoch]:.4f},{val_errors[epoch]:.4f}\n")
# 绘制训练和验证损失及误差图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_errors, label='Training Error')
plt.plot(range(1, num_epochs + 1), val_errors, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.title('Training and Validation Error')
plt.tight_layout()
filename = "training_validation_curves.png"
filepath = os.path.join(out_dir, filename)
plt.savefig(filepath)
plt.show()

# 测试模型
test_loss = 0.0
test_error = 0.0

# 打开文件用于保存第一个样本的测试结果

# 创建一个新的文件，用于保存每个样本的总误差
error_filename = "total_errors.txt"  # 保存总误差的文件
error_filepath = os.path.join(out_dir, error_filename)


with torch.no_grad():  # 不需要计算梯度
    sample_id = 0
    with open(error_filepath, 'w') as error_file:  # 打开总误差文件以便写入

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            stress_output, phase_output = model(inputs)
            # 计算两个输出的损失
            loss_stress = criterion(stress_output, targets[:, :30])
            loss_phase = criterion(phase_output, targets[:, 30:])
            test_loss += loss_stress.item() + loss_phase.item()

            # 计算测试集的误差
            error_stress = torch.abs(stress_output - targets[:, :30]).mean()
            error_phase = torch.abs(phase_output - targets[:, 30:]).mean()
            test_error += (error_stress + error_phase).item()

            # 遍历batch中每个样本
            for i in range(inputs.size(0)):
                if sample_id >= 200:
                    break  # 只写前10个样本
                true_stress = targets[i, :30].cpu().numpy()
                pred_stress = stress_output[i].cpu().numpy()
                true_phase = targets[i, 30:].cpu().numpy()
                pred_phase = phase_output[i].cpu().numpy()
                # 计算当前样本的误差
                error_stress_sample = (stress_output[i] - targets[i, :30]).mean().item()  # 不取绝对值
                error_phase_sample = (phase_output[i] - targets[i, 30:]).mean().item()  # 不取绝对值
                # 保存每个样本的预测结果到独立的txt文件
                sample_filename = f"test_sample_{sample_id + 1}.txt"
                sample_filepath = os.path.join(out_dir, sample_filename)

                with open(sample_filepath, 'w') as f:
                    f.write("Sample,True Stress,Predicted Stress,True Phase,Predicted Phase\n")  # 写入表头
                    for j in range(30):  # 每个样本写30个数据点
                        f.write(f"{sample_id + 1},{j + 1},{true_stress[j]:.4f},{pred_stress[j]:.4f},{true_phase[j]:.4f},{pred_phase[j]:.4f}\n")
                    f.write("\n")  # 每个样本之间空一行
                    f.write(f"Total Error: {error_stress_sample + error_phase_sample:.4f}\n")  # 写入总误差
                sample_id += 1
                error_file.write(f"Sample {sample_id + 1}: Total Error: {error_stress_sample + error_phase_sample:.4f}, "
                                 f"Phase Output Last Value: {pred_phase[-1]:.4f}, Target Phase Last Value: {true_phase[-1]:.4f}\n")
            if sample_id >= 200:
                break  # 外层也跳出

# 计算测试集的平均损失和误差
avg_test_loss = test_loss / len(test_loader)
avg_test_error = test_error / len(test_loader)

# 打印测试结果
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Error: {avg_test_error:.4f}")



train_loss = 0.0
train_error = 0.0
# 打开文件用于保存训练集的预测结果
textname = "train_results.txt"  # 输出每个样本的文件名
filepathtext = os.path.join(out_dir, textname)
with open(filepathtext, 'w') as f:
    f.write("Sample,True Stress,Predicted Stress,True Phase,Predicted Phase\n")  # 写入表头
    sample_id = 0
    model.eval()  # 切换到评估模式，确保不会进行梯度计算
    with torch.no_grad():  # 不需要计算梯度
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            stress_output, phase_output = model(inputs)

            # 计算两个输出的损失
            loss_stress = criterion(stress_output, targets[:, :30])
            loss_phase = criterion(phase_output, targets[:, 30:])
            train_loss += loss_stress.item() + loss_phase.item()

            # 计算训练集的误差
            error_stress = torch.abs(stress_output - targets[:, :30]).mean()
            error_phase = torch.abs(phase_output - targets[:, 30:]).mean()
            train_error += (error_stress + error_phase).item()

            # 遍历batch中每个样本
            for i in range(inputs.size(0)):
                if sample_id >= 32:
                    break  # 只写前32个样本
                true_stress = targets[i, :30].cpu().numpy()
                pred_stress = stress_output[i].cpu().numpy()
                true_phase = targets[i, 30:].cpu().numpy()
                pred_phase = phase_output[i].cpu().numpy()

                for j in range(30):  # 每个样本写30个数据点
                    f.write(
                        f"{sample_id + 1},{j + 1},{true_stress[j]:.4f},{pred_stress[j]:.4f},{true_phase[j]:.4f},{pred_phase[j]:.4f}\n")
                f.write("\n")  # 每个样本之间空一行
                sample_id += 1

            if sample_id >= 32:
                break  # 外层也跳出

# 计算训练集的平均损失和误差
avg_train_loss = train_loss / len(train_loader)
avg_train_error = train_error / len(train_loader)

# 打印训练结果
print(f"Train Loss: {avg_train_loss:.4f}")
print(f"Train Error: {avg_train_error:.4f}")
