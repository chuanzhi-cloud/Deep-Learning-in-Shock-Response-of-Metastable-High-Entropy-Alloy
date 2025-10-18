# 创建模型包含生成器判别器
# 创建模型包含生成器判别器
# 创建模型包含生成器判别器
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.nn as nn


# 生成器 U-Net（输入照片为64*64）
class Generator(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, ngf=64):
        """
        定义生成器的网络结构
        :param in_ch: 输入数据的通道数
        :param out_ch: 输出数据的通道数
        :param ngf: 第一层卷积的通道数 number of generator's first conv filters
        """
        super(Generator, self).__init__()
        # 下面的激活函数都放在下一个模块的第一步 是为了skip-connect方便
        # Initial layer
        self.input_layer = nn.Sequential(
            nn.Linear(30, 4 * 32 * 32),
            nn.Unflatten(1, (4, 32, 32))
        )
        # 左半部分 U-Net encoder
        # 每层输入大小折半，从输入图片大小64开始
        self.en0 = nn.Sequential(
            nn.Conv2d(in_ch, ngf, kernel_size=3, stride=1, padding=1),
            # 输入图片已正则化 不需BatchNorm
        )
        # 1 * 32*32（输入）
        self.en1 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # 输入图片已正则化 不需BatchNorm
        )
        # 64 * 32*32
        self.en2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        # 128 * 32*32
        self.en3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        # 256 * 16*16
        self.en4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 512 * 8*8
        self.en5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 512 * 4*4
        self.en6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        # 512 * 2*2
        self.en7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf * 8)
            # 512 * 1*1
        )
        # 右半部分 U-Net decoder
        # skip-connect: 前一层的输出+对称的卷积层
        # 512*1 * 1（输入）
        self.de1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8),
            # nn.Dropout(p=0.5)
        )
        # 512*2 * 2
        self.de2 = nn.Sequential(
            nn.ReLU(inplace=True),
            # skip-connect 所以输入管道数是之前输出的2倍
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            # nn.Dropout(p=0.5)
        )
        # 512*4 * 4
        self.de3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            # nn.Dropout(p=0.5)
        )
        # 512 *8 * 8
        self.de4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            # nn.Dropout(p=0.5)
        )
        # 128*16 * 16
        self.de5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            # nn.Dropout(p=0.5)
        )
        # 64 *32 * 32
        self.de6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            # nn.Dropout(p=0.5)
        )
        # 64 * 32 * 32
        self.de7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            # 3* 32*32
        )

    def forward(self, X):
        """
        生成器模块前向传播
        :param X: 输入生成器的数据
        :return: 生成器的输出
        """
        # Encoder
        X = self.input_layer(X)
        en0_out = self.en0(X)
        en1_out = self.en1(en0_out)
        en2_out = self.en2(en1_out)  # 1283232
        en3_out = self.en3(en2_out)  # 2561616
        en4_out = self.en4(en3_out)  # 51288
        en5_out = self.en5(en4_out)  # 51244
        en6_out = self.en6(en5_out)  # 51222
        en7_out = self.en7(en6_out)  # 51211

        # Decoder
        de1_out = self.de1(en7_out)  # 51222
        de1_cat = torch.cat([de1_out, en6_out], dim=1)  # 102422
        de2_out = self.de2(de1_cat)  # 51244
        de2_cat = torch.cat([de2_out, en5_out], 1)  # 102444
        de3_out = self.de3(de2_cat)  # 51288
        de3_cat = torch.cat([de3_out, en4_out], 1)  # 102488
        de4_out = self.de4(de3_cat)  # 2561616
        de4_cat = torch.cat([de4_out, en3_out], 1)  # 5121616
        de5_out = self.de5(de4_cat)  # 1283232
        de5_cat = torch.cat([de5_out, en2_out], 1)  # 2563232
        de6_out = self.de6(de5_cat)  # 643232
        de7_out = self.de7(de6_out)

        return de7_out


# # Test the model
# if __name__ == "__main__":
#     model = Generator()
#     input_tensor = torch.randn(1, 4, 64, 64)  # 输入为包含30个元素的向量
#     output = model(input_tensor)
#     print("Output shape:", output.shape)
# # 测试代码
# from torchsummary import summary
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# netG = Generator().to(device)
# netD = Discriminator().to(device)
# # 测试生成器输出形状
# noise = torch.randn(16, nz, 1, 1, device=device)  # Batch size = 16, nz = 100
# summary(netG, (30,))
# summary(netD, (4, 30, 30))
# fake_images = netG(noise)
# print("Generator output shape:", fake_images.shape)  # 期望输出: (16, nc, 6, 6)
# # 测试判别器输出形状
# output = netD(torch.randn(64, 4, 30, 30, device=device))
# print("Discriminator output shape:", output.size())  # 期望输出: (16, 1, 1, 1)
# 测试模型输出形状
# model = Generator()
# input_tensor = torch.randn(1, 30)  # 输入为包含30个元素的向量
# 前向传播并打印输出形状
# output = model(input_tensor)
# print("模型输出形状:", output.shape)
# 辨别器 PatchGAN（其实就是卷积网络而已） ##
class Discriminator(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, ndf=32, response_dim=30):
        """
        定义判别器的网络结构
        :param in_ch: 输入数据的通道数
        :param ndf: 第一层卷积的通道数 number of discriminator's first conv filters
        """
        super(Discriminator, self).__init__()
        # 不是输出一个表示真假概率的实数，而是一个N*N的Patch矩阵（此处为30*30），其中每一块对应输入数据的一小块
        # in_ch + out_ch 是为将对应真假数据同时输入
        # 64 * 64（输入）
        # 输入 response 转换层
        self.response_to_image = nn.Sequential(
            nn.Linear(response_dim, 4 * 32 * 32),
            nn.Unflatten(1, (4, 32, 32))
        )
        # PatchGAN 主网络
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, ndf, kernel_size=3, stride=1, padding=1),
            # 输入图片已正则化 不需BatchNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64 * 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32 * 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # 30 * 30（输出的Patch大小）

    def forward(self, response, target):
        """
        判别器模块正向传播
        :param X: 输入判别器的数据
        :return: 判别器的输出
        """
        response_image = self.response_to_image(response)
        combined_input = torch.cat([response_image, target], dim=1)
        layer1_out = self.layer1(combined_input)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        return layer5_out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)


# from torchsummary import summary
# # 定义 Discriminator 模型示例
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Discriminator().to(device)
# # 检查模型的每层输入输出形状，输入形状为 8 个通道，图像大小为 64x64
# summary(model, [(30,), (4, 32, 32)])
# 创建模型包含生成器判别器
# 创建模型包含生成器判别器
# 创建模型包含生成器判别器
# =========================================================================
# ================================加载数据==================================
# =========================================================================
# Set random seed for reproducibility
import random
import xlrd
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
import numpy as np


# Root directory for dataset
# 标准化到 [-1, 1] 范围
def normalize_to_minus1_1(data):
    return 2 * data - 1


# 读取 responsetotal.xls 数据
filename2 = '应力应变曲线.xls'  # 替换为你的文件路径
workbook2 = xlrd.open_workbook(filename2)
sheet = workbook2.sheet_by_index(0)
response0 = np.zeros((sheet.nrows, 30))
for row_idx in range(sheet.nrows):
    row_data = np.array(sheet.row_values(row_idx))
    response0[row_idx, :] = row_data[:] / 1900
noise = np.random.normal(0, 0.05, response0.shape)  # 添加少量高斯噪声
response0 += noise
response0 = normalize_to_minus1_1(response0)
# We can use an image folder dataset the way we have it setup.
# Create the dataset
from torchvision import datasets, transforms
from PIL import Image
import os
import re
dataroot = "F:\\Work\\2025\\Direct and inverse\\所有的图\\case study\\train\\output"  # 自定义加载函数，确保加载为 RGBA 格式


def load_rgba_image(path):
    return Image.open(path).convert("RGBA")


# 数据变换与加载
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # 将图像转换为 [C, H, W] 张量
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
])


# 自定义排序函数：自然排序
def natural_sort_key(file):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file)]


# 按文件名自然排序加载图片
image_files = sorted(
    [os.path.join(root, file)
     for root, _, files in os.walk(dataroot)
     for file in files if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))],
    key=natural_sort_key
)
# 加载图片到列表中
targets = torch.stack([transform(load_rgba_image(path)) for path in image_files])

# 确保形状匹配
features = torch.from_numpy(response0.astype(np.float32))  # 这里没问题，接下来继续检查输出
train_dataset = torch.utils.data.TensorDataset(features, targets)  # 只是加载了数据集，对训练没有影响
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True  # 从数据集中根据batch取数据，每次取得时候都将数据打乱
)
# 上面个的加载程序是正确的，每一行的响应就对应着顺序为他的图片
# 上面个的加载程序是正确的，每一行的响应就对应着顺序为他的图片
# 上面个的加载程序是正确的，每一行的响应就对应着顺序为他的图片
# 上面个的加载程序是正确的，每一行的响应就对应着顺序为他的图片
# 上面个的加载程序是正确的，每一行的响应就对应着顺序为他的图片
# 上面个的加载程序是正确的，每一行的响应就对应着顺序为他的图片
# ========================================================================
# ================================加载数据==================================
# =========================================================================

# =========================================================================
# ================================模型训练==================================
# =========================================================================
# 保存每次测试的四个样本
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# Root directory for dataset
out_dir = os.path.join('./', datetime.datetime.now().strftime('%Y%m%d-%H%M') + "out")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir)  # 创建一个SummaryWriter的示例，默认目录名字为runs   模型的可视化训练日志存在logs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 下面的程序测试从总的数据集train_dataset中随机拿出来16个，第一次保存真实的图片，后面一直使用这16个进行测试
# 从 DataLoader 中取出一个批次数据
# 提取固定样本（手动选择固定索引）
fixed_indices = list(range(1984, 2000))  # 固定选择前 16 个样本（可根据需要调整索引）
# print(fixed_indices)
fixed_noise = features[fixed_indices].to(device)  # 固定输入特征
fixed_targets = targets[fixed_indices]  # 固定目标图像
# 假设16个样本的大小为 (4, 32, 32)
# 初始化一个 numpy 数组来存储所有样本
all_images = []
for idx in range(16):
    fix_target = fixed_targets[idx, :]  # 获取第 idx 个样本
    image = (fix_target + 1) / 2  # 将范围调整到 [0, 1]
    image = np.transpose(image, (1, 2, 0)).cpu().numpy()  # 转换为 H, W, C 格式
    all_images.append(image)
# 将所有样本组合为一个网格，网格中每张图片之间增加空隙
rows, cols = 4, 4  # 将16张图片放在4x4的网格中
img_size = 32  # 每张图片大小
padding = 4  # 每张图片之间的空隙大小
grid_height = rows * img_size + (rows - 1) * padding
grid_width = cols * img_size + (cols - 1) * padding
# 初始化网格，填充为全白（1.0）
grid_image = np.ones((grid_height, grid_width, 4))  # (H, W, C)
# 将每张图片放到网格中
for idx, image in enumerate(all_images):
    row = idx // cols
    col = idx % cols
    start_y = row * (img_size + padding)
    start_x = col * (img_size + padding)
    grid_image[start_y:start_y + img_size, start_x:start_x + img_size, :] = image
# 保存网格为 TIFF 图像
filename = "grid_image_with_padding.tiff"
filepath = os.path.join(out_dir, filename)
plt.imsave(filepath, grid_image, format="tiff")
# 保存每个样本为单独的文本文件
coefficients = [360, 180, 360, 400]
for idx, image in enumerate(all_images):
    textname = f"image_{idx + 1}.txt"  # 输出每个样本的文件名
    filepathtext = os.path.join(out_dir, textname)
    with open(filepathtext, "w") as f:
        f.write("Phase X Y Euler1 Euler2 Euler3 Temperature\n")
        image_text = image.reshape(1024, 4).T  # 转换为 (4, 1024)
        for global_row_index in range(1024):
            row_data = image_text[:, global_row_index]
            modified_row_data = [row_data[k] * coefficients[k] for k in range(len(coefficients))]
            line = [1, global_row_index + 1, global_row_index + 1] + [f"{value:.2f}" for value in modified_row_data]
            f.write(" ".join(map(str, line)) + "\n")
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# 这上面个对于前16个样本从数据集中拿出来保存成一个大图的顺序没问题，继续保存成16txt文件出来的数据也没问题。
# =========================================================================
# 定义训练参数
lamb = 100  # 在生成器的目标函数中L1正则化的权重
# 声明生成器、判别器 及相应的初始化
netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)
# 目标函数 & 优化器
import torch.optim as optim
import torchvision
from pytorch_msssim import ssim

def calculate_ssim(target_images, generated_images):
    ssim_value = ssim(target_images, generated_images, data_range=1.0, size_average=True)
    return ssim_value.item()
criterion = nn.BCELoss()  # 二分类的损失函数
L1 = nn.L1Loss()  # Pix2Pix论文中在传统GAN目标函数加上了L1
optimizer_G = optim.Adam(netG.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.00008, betas=(0.5, 0.999))

num_epochs = 500
# 损失变量
xiangduiwucha = []

jueduiwucha = []
L1wucha = []
L2wucha = []
xiangsixingwucha = []
# Training Loop
print("Starting Training Loop...")
# For each epoch
D_Loss, G_Loss, L1_Loss = [], [], []
for epoch in range(num_epochs):
    total_D_grad_norm = 0
    total_G_grad_norm = 0
    count_D_params = 0
    count_G_params = 0
    # For each batch in the dataloader
    for i, (input_stress, target_texture) in enumerate(train_loader, 0):
        D_losses, G_losses, L1_losses, d_l, g_l, my_l1 = [], [], [], 0, 0, 0
        input_stress = input_stress.to(device)
        # 2.响应对应的目标织构
        target_texture = target_texture.to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        netD.zero_grad()
        # 在真实样本上
        disc_real = netD(input_stress, target_texture).squeeze()  # .view(-1)
        D_x = disc_real.mean().item()
        real_label = torch.ones(disc_real.size()).to(device)
        disc_real_loss = criterion(disc_real, real_label)
        # 在假数据上
        # Forward pass real batch through D
        fake_output = netG(input_stress)
        disc_fake = netD(input_stress, fake_output).squeeze()
        fake_label = torch.zeros(disc_fake.size()).to(device)
        disc_fake_loss = criterion(disc_fake, fake_label)
        # 反向传播并优化
        D_loss = (disc_real_loss + disc_fake_loss) * 0.5
        D_loss.backward()
        # 计算判别器的梯度范数
        for param in netD.parameters():
            if param.grad is not None:
                total_D_grad_norm += param.grad.data.norm(2).item()
                count_D_params += 1
        optimizer_D.step()
        D_losses.append(D_loss.item())
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        fake_output = netG(input_stress)
        disc_fake = netD(input_stress, fake_output).squeeze()
        real_label = torch.ones(disc_fake.size()).to(device)
        G_BCE_loss = criterion(disc_fake, real_label)
        G_L1_Loss = L1(fake_output, target_texture)
        # 反向传播并优化
        G_loss = G_BCE_loss + lamb * G_L1_Loss
        G_loss.backward()
        # 计算生成器的梯度范数
        for param in netG.parameters():
            if param.grad is not None:
                total_G_grad_norm += param.grad.data.norm(2).item()
                count_G_params += 1
        optimizer_G.step()
        G_losses.append(G_loss.item())
        L1_losses.append(G_L1_Loss.item())
        d_l, g_l, my_l1 = np.array(D_losses).mean(), np.array(G_losses).mean(), np.array(L1_losses).mean()
        # print(d_l, g_l, my_l1)
        # Output training stats
        if i % 50 == 0:
            print('[%d / %d]: loss_d= %.3f  loss_g= %.3f Loss_L1= %.3f D_x= %.3f' %
                  (epoch + 1, num_epochs, d_l, g_l, my_l1, D_x))

    D_Loss.append(d_l)
    G_Loss.append(g_l)
    L1_Loss.append(my_l1)
    avg_D_grad_norm = total_D_grad_norm / count_D_params if count_D_params > 0 else 0
    avg_G_grad_norm = total_G_grad_norm / count_G_params if count_G_params > 0 else 0
    writer.add_scalar("Gradient/Discriminator", avg_D_grad_norm, epoch)
    writer.add_scalar("Gradient/Generator", avg_G_grad_norm, epoch)
    writer.add_scalar('data/lossD', d_l, epoch)
    writer.add_scalar('data/lossG', g_l, epoch)
    writer.add_scalar('lossmy_l1', my_l1, epoch)
    writer.add_scalar("DX", D_x, epoch)
    if (epoch + 1) % 5 == 0 or (epoch == num_epochs - 1):
        with torch.no_grad():
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fake_outputs = netG(fixed_noise).detach().cpu()  # 使用16个固定输入生成
            fake_images = (fake_outputs + 1) / 2  # 将范围调整到 [0, 1]
            # 取对应的真实目标图
            target_images = fixed_targets  # 提取对应的16个真实目标
            target_images = (torch.Tensor(target_images) + 1) / 2  # 将范围调整到 [0, 1]
            # 分别构建真实图和生成图的网格
            target_grid = torchvision.utils.make_grid(target_images, nrow=4, padding=10, normalize=False)  # 每行4张图
            fake_grid = torchvision.utils.make_grid(fake_images, nrow=4, padding=10, normalize=False)  # 每行4张图
            # 拼接真实图和生成图：左右拼接
            combined_grid = torch.cat((target_grid, fake_grid), dim=2)  # 按宽度方向拼接
            combined_image = np.transpose(combined_grid.numpy(), (1, 2, 0))  # 转换形状为 (H, W, C)
            # 保存拼接的大图
            grid_filename = f"comparison_grid_{current_time}.tiff"  # 输出文件名
            grid_filepath = os.path.join(out_dir, grid_filename)
            plt.imsave(grid_filepath, combined_image, format="tiff")  # 保存网格图片
            # —— 新增：逐张保存生成的16张假图 —— #
            for idx, tar_img in enumerate(target_images, start=1):
                # fake_img 是 Tensor，shape (4,32,32)，值在 [0,1]
                img_np = np.transpose(tar_img.numpy(), (1, 2, 0))  # 转成 (H, W, C)
                single_name = f"real{idx}.tiff"
                single_path = os.path.join(out_dir, single_name)
                plt.imsave(single_path, img_np, format="tiff")
            # —— End 新增 —— #
            # —— 新增：逐张保存生成的16张假图 —— #
            for idx, fake_img in enumerate(fake_images, start=1):
                # fake_img 是 Tensor，shape (4,32,32)，值在 [0,1]
                img_np = np.transpose(fake_img.numpy(), (1, 2, 0))  # 转成 (H, W, C)
                single_name = f"predict{idx}.tiff"
                single_path = os.path.join(out_dir, single_name)
                plt.imsave(single_path, img_np, format="tiff")
            # —— End 新增 —— #
            # 计算误差
            l1_loss = torch.nn.functional.l1_loss(fake_images, target_images)
            l2_loss = torch.nn.functional.mse_loss(fake_images, target_images)
            ssim_value = calculate_ssim(target_images, fake_images)

            absolute_errors = torch.abs(target_images - fake_images)
            relative_errors = torch.abs(target_images - fake_images) / (torch.abs(target_images) + 1e-8)  # 防止除零
            mean_absolute_error = absolute_errors.mean().item()
            mean_relative_error = relative_errors.mean().item()
            L1wucha.append(l1_loss)
            L2wucha.append(l2_loss)
            xiangsixingwucha.append(ssim_value)
            xiangduiwucha.append(mean_relative_error)
            jueduiwucha.append(mean_absolute_error)
            writer.add_scalar('L1wucha', l1_loss, epoch)
            writer.add_scalar('L2wucha', l2_loss, epoch)
            writer.add_scalar('xiangsixingwucha', ssim_value, epoch)
            writer.add_scalar('xiangduiwucha', mean_relative_error, epoch)
            writer.add_scalar('jueduiwucha', mean_absolute_error, epoch)
            # 分别保存生成图的数值数据
            coefficients = [360, 180, 360, 400]
            for idx, fake_image in enumerate(fake_images):
                image = fake_image.numpy()  # 转换为numpy数组
                image = np.transpose(image, (1, 2, 0))  # 转换形状为 (H, W, C)
                # 保存单张图片的数值数据到文本文件
                textname = f"generated_{current_time}_{idx + 1}.txt"  # 输出文本文件名
                textpath = os.path.join(out_dir, textname)
                with open(textpath, "w") as f:
                    # 写入列名行
                    f.write("Phase X Y Euler1 Euler2 Euler3 Temperature\n")
                    # 提取张量的切片（4, 32, 32），然后将每个 32x32 的切片展平为 1024 行
                    image_text = image.reshape(1024, 4).T  # 将 (4, 32, 32) 切片展平为 (4, 1024)
                    # 写入1024行，每行包含 3 列索引和 4 列数据
                    for global_row_index in range(1024):
                        # 每行取四组数据中对应的全局行索引位置的数据
                        row_data = image_text[:, global_row_index]  # 从每一组的对应位置取值，形成一行4个数据
                        modified_row_data = [row_data[k] * coefficients[k] for k in range(len(coefficients))]
                        line = [1, global_row_index + 1, global_row_index + 1] + [f"{value:.2f}" for value in
                                                                                  modified_row_data]
                        # 写入文件
                        f.write(" ".join(map(str, line)) + "\n")
            # 保存模型
            torch.save(netG.state_dict(), out_dir + f"./{current_time}netG.pt")

writer.close()  # 将event log写完之后,close()
# 画出loss图
filepath = os.path.join(out_dir, 'loss.png')
plt.plot(np.arange(1, num_epochs + 1), D_Loss, label='Discriminator Losses')
plt.plot(np.arange(1, num_epochs + 1), np.array(G_Loss) / float(lamb), label='Generator Losses / 100')
plt.legend()
plt.savefig(filepath)
# plt.show()

# 保存相关的数据
filepath = os.path.join(out_dir, "loss.txt")
f = open(filepath, "w")  # 看上面的保存格式，每个epoch保存一次
f.write("Epoch G_loss D_Loss L1_loss\n")
for i in range(num_epochs):
    # print(G_Loss[i],D_Loss[i],L1_Loss[i])
    f.write(str(i + 1) + "    " + str(G_Loss[i]) + "    " + str(D_Loss[i]) + "    " + str(L1_Loss[i]) + "\n")
f.close()
filepath = os.path.join(out_dir, "wucha.txt")
f = open(filepath, "w")  # 看上面的保存格式，区分每个iteration保存一次还是一个epoch保存一次
for i in range(len(xiangduiwucha)):
    f.write(str(i + 1) + "    " + str(xiangduiwucha[i]) + "    " + str(jueduiwucha[i]) + "\n")
f.close()
