import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os

# 假设你有一个CustomDataset类，用于加载CSV和图片
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None):
        """
        初始化函数，用于读取CSV文件和设置图片文件夹路径及变换

        参数:
        csv_file (str): CSV文件的路径
        img_folder (str): 存放图片的文件夹路径
        transform (callable, optional): 一个用于对图片进行变换的可调用对象
        """
        # 尝试使用不同的编码读取CSV文件
        try:
            self.data = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_file, encoding='gbk')  # 如果utf-8编码失败，尝试使用gbk编码
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本

        参数:
        idx (int): 数据集中的索引

        返回:
        tuple: 一个包含参数和图片的元组
        """
        # 读取CSV中的数据
        row = self.data.iloc[idx]
        img_name = os.path.join(self.img_folder, f"{idx+1}.png")  # 假设图片名为1.png, 2.png等
        image = Image.open(img_name)
        
        # 读取图片和对应的51个参数
        image = image.convert("RGB")  # 确保图片是3通道的
        parameters = torch.tensor(row.values, dtype=torch.float32)  # 将CSV中的51个参数转换为浮点张量
        
        # 如果定义了变换，则对图片应用变换
        if self.transform:
            image = self.transform(image)

        return parameters, image  # 返回参数和图片

# 定义模型
class ImageGenerationCNN(nn.Module):
    def __init__(self):
        """
        初始化ImageGenerationCNN模型，这是一个用于生成图像的卷积神经网络模型
        """
        super(ImageGenerationCNN, self).__init__()
        # 第一层全连接层，将输入的51个特征映射到8192个特征
        self.fc1 = nn.Linear(51, 8192)
        # 第一层卷积层，输入通道为8，输出通道为64，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        # 第二层卷积层，输入通道为64，输出通道为128，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 第三层卷积层，输入通道为128，输出通道为256，卷积核大小为3x3，填充为1
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # 第一层上采样卷积层，输入通道为256，输出通道为128，卷积核大小为4x4，步长为2，填充为1
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)       
        # 第二层上采样卷积层，输入通道为128，输出通道为64，卷积核大小为4x4，步长为2，填充为1
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # 第三层上采样卷积层，输入通道为64，输出通道为3（对应RGB图像的3个颜色通道），卷积核大小为4x4，步长为2，填充为1
        self.upconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Tanh激活函数，用于生成图像，将输出值范围限制在[-1, 1]
        self.tanh = nn.Sigmoid()
    def forward(self, x):
        """
        定义数据通过网络的前向传播过程

        参数:
        x (Tensor): 输入张量，形状为[batch_size, 51]

        返回:
        Tensor: 输出张量，形状为[batch_size, 3, 180, 180]
        """
        # 应用全连接层fc1，将输入的51个特征映射到8192个特征
        x = self.fc1(x)
        
        # 将全连接层的输出（8192个特征）重塑为4维张量，形状为[batch_size, 8, 32, 32]
        # 这里假设8192 = 8 * 32 * 32，即每个批次的大小是8192，并重塑为8个通道，每个通道32x32的图像
        x = x.view(x.size(0), 8, 32, 32)

        # 应用第一层卷积层，并使用ReLU激活函数
        x = self.relu(self.conv1(x))
        
        # 应用第二层卷积层，并使用ReLU激活函数
        x = self.relu(self.conv2(x))
        
        # 应用第三层卷积层，并使用ReLU激活函数
        x = self.relu(self.conv3(x))

        # 应用第一层上采样卷积层（反卷积），并使用ReLU激活函数，输出尺寸变为64x64
        x = self.relu(self.upconv1(x))
        
        # 应用第二层上采样卷积层（反卷积），并使用ReLU激活函数，输出尺寸变为128x128
        x = self.relu(self.upconv2(x))
        
        # 应用第三层上采样卷积层（反卷积），并使用Tanh激活函数，输出尺寸变为256x256
        x = self.tanh(self.upconv3(x))

        # 使用双线性插值调整输出图像的尺寸到180x180
        x = F.interpolate(x, size=(180, 180), mode='bilinear', align_corners=False)

        # 返回调整尺寸后的输出张量
        return x

# 数据加载器
csv_file = 'zhiJing.csv'
img_folder = 'picture'
batch_size = 1

# 假设你有合适的图像变换
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(csv_file, img_folder, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 假设你已经确保DataLoader的batch_size一致

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageGenerationCNN().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式

    for i, (inputs, targets) in enumerate(train_loader):
        # 将输入数据和目标数据转移到正确的设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 打印输入和目标的尺寸，用于调试
        # print(f"Epoch {epoch+1}, Step {i+1}")
        # print(f"Inputs size: {inputs.size()}")
        # print(f"Targets size: {targets.size()}")

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 打印输出尺寸，用于调试
        print(f"Outputs size: {outputs.size()}")

        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()

        # 优化步骤
        optimizer.step()

        # 打印日志
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # 每个epoch保存模型
    if epoch==num_epochs-1:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
