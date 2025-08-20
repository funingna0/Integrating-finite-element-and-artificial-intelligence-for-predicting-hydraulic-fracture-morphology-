import torch
import torch.nn as nn

class ImageGenerationCNN(nn.Module):
    def __init__(self):
        super(ImageGenerationCNN, self).__init__()

        # 图像生成分支
        self.fc1 = nn.Linear(54, 8192)
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

        # 浮点数生成分支
        self.float_fc = nn.Linear(54, 128)
        self.float_out = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        # 图像生成路径
        img_x = self.fc1(x)
        img_x = img_x.view(img_x.size(0), 8, 32, 32)
        img_x = self.relu(self.conv1(img_x))
        img_x = self.relu(self.conv2(img_x))
        img_x = self.relu(self.conv3(img_x))
        img_x = self.relu(self.upconv1(img_x))
        img_x = self.relu(self.upconv2(img_x))
        img_x = self.tanh(self.upconv3(img_x))
        img_x = torch.nn.functional.interpolate(img_x, size=(256, 256), mode='bilinear', align_corners=False)

        # 浮点数生成路径
        float_x = self.relu(self.float_fc(x))
        float_out = self.float_out(float_x)

        return img_x, float_out
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGenerationCNN0(nn.Module):
    def __init__(self):
        super(ImageGenerationCNN0, self).__init__()
        
        # 全连接层，用于将输入特征映射到更高的维度
        self.fc1 = nn.Linear(54, 2048)  # 输入54个特征
        self.fc2 = nn.Linear(2048, 16384)  # 更大的一层，用于更细致的特征提取
        
        # 第一组卷积层，逐步提取更高层次的特征
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # 上采样层，将特征图大小逐步放大
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
        # 激活函数和归一化
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)  # LeakyReLU
        self.tanh = nn.Tanh()  # 输出为 [-1, 1]
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        # 通过全连接层，获得初步的特征向量
        x = self.fc1(x)
        # print(x.size())
        x = self.relu(x)
        x = self.fc2(x)
        # print(x.size())

        x = self.relu(x)
        
        # 调整特征图的形状，准备卷积操作
        x = x.view(x.size(0), 1, 128, 128)
        # print(x.size())
        # 逐层卷积，提取特征
        x = self.relu(self.conv1(x))
        # print(x.size())
        x = self.relu(self.conv2(x))
        # print(x.size())
        x = self.relu(self.conv3(x))
        # print(x.size())
        x = self.relu(self.conv4(x))
        # print(x.size())
        x = self.relu(self.conv5(x))
        # print(x.size())
        
        # 上采样阶段，逐步恢复图像大小
        x = self.relu(self.upconv1(x))
        # print(x.size())
        x = self.relu(self.upconv2(x))
        # print(x.size())
        x = self.relu(self.upconv3(x))
        # print(x.size())
        x = self.relu(self.upconv4(x))
        # print(x.size())
        x = self.tanh(self.upconv5(x))
        # print(x.size())
        
        # 最后一步，通过双线性插值将输出的图像大小调整为 256x256
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        # print(x.size())
        #参数规模50,331,648
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGenerationCNN1(nn.Module):
    def __init__(self):
        super(ImageGenerationCNN1, self).__init__()
        
        # 全连接层，用于将输入特征映射到更高的维度
        self.fc1 = nn.Linear(54, 2048)  # 输入54个特征
        self.fc2 = nn.Linear(2048, 16384)  # 更大的一层，用于更细致的特征提取

        # 第一组卷积层，逐步提取更高层次的特征
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # 上采样层，将特征图大小逐步放大
        self.upconv1 = nn.ConvTranspose2d(512, 64, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        # self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # self.upconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
        # 激活函数和归一化
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)  # LeakyReLU
        self.relu = nn.ReLU(inplace=False)  # 使用 inplace=False
        self.sigmoid = nn.Sigmoid()
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.fc_bn1 = nn.BatchNorm1d(2048)
        self.fc_bn2 = nn.BatchNorm1d(16384)

    def forward(self, x):
        # 通过全连接层，获得初步的特征向量
        x = self.fc1(x)
        # print(x.size())
        x = self.fc_bn1(x)  # 批归一化
        x = self.relu(x)  # 使用非-inplace ReLU
        
        x = self.fc2(x)
        x = self.fc_bn2(x)  # 批归一化
        x = self.relu(x)  # 使用非-inplace ReLU

        # 调整特征图的形状，准备卷积操作
        x = x.view(x.size(0), 1, 128, 128)

        # 卷积层，加入批归一化和ReLU
        x = self.bn1(self.conv1(x))
        x = self.leaky_relu(x)
        x = self.bn2(self.conv2(x))
        x = self.leaky_relu(x)
        x = self.bn3(self.conv3(x))
        x = self.leaky_relu(x)
        x = self.bn4(self.conv4(x))
        x = self.leaky_relu(x)
        x = self.bn5(self.conv5(x))
        x = self.leaky_relu(x)
        
        # 残差连接（ResNet结构）
        x = self.upconv1(x)
        x = self.relu(x)
        # print(x.size())
        # print(residual.size())
        residual = x  # 保存残差
        # 残差连接
        x = x + residual  # 使用非-inplace 方式进行加法操作

        # 上采样阶段，逐步恢复图像大小
        x = self.upconv2(x)
        # print(x.size())
        # x = self.relu(x)
        # residual = x  # 更新残差
        # x = x + residual  # 使用非-inplace 方式进行加法操作
        
        # x = self.upconv3(x)
        # x = self.relu(x)
        # residual = x  # 更新残差
        # x = x + residual  # 使用非-inplace 方式进行加法操作
        
        # x = self.upconv4(x)
        # x = self.relu(x)
        # residual = x  # 更新残差
        # x = x + residual  # 使用非-inplace 方式进行加法操作

        # # 输出图像的大小
        # x = self.upconv5(x)
        x = self.sigmoid(x)

        # 最后一步，通过双线性插值将输出的图像大小调整为 256x256
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x

class ImageGenerationCNNSmall(nn.Module):
    def __init__(self):
        super(ImageGenerationCNNSmall, self).__init__()

        # 修改为8192个特征
        self.fc1 = nn.Linear(54, 8192)
        self.conv1 = nn.Conv2d(8, 64, kernel_size=3, padding=1)  # 输入8通道
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # fc1 之后的输出，确保批次大小一致
        x = self.fc1(x)
        
        # 将fc1的输出（8192个特征）重塑为 [batch_size, 8, 32, 32]
        x = x.view(x.size(0), 8, 32, 32)

        # 卷积层
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # 反卷积（上采样）操作
        x = self.relu(self.upconv1(x))  # 64x64
        x = self.relu(self.upconv2(x))  # 128x128
        x = self.tanh(self.upconv3(x))  # 256x256

        # 将输出调整为180x180
        x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x
    
class BPNeuralNetwork(nn.Module):
    def __init__(self):
        super(BPNeuralNetwork, self).__init__()
        
        # 定义三层隐藏层和输出层
        self.fc1 = nn.Linear(54, 718)  # 第一层，输入54个特征，输出718个特征
        self.fc2 = nn.Linear(718, 2048)  # 第二层
        self.fc3 = nn.Linear(2048, 512)  # 第三层
        self.fc4 = nn.Linear(512, 1)  # 输出层，生成一个浮点数

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层
        x = self.relu(self.fc2(x))  # 第二层
        x = self.relu(self.fc3(x))  # 第三层
        x = self.fc4(x)  # 输出层，不需要激活函数，直接输出浮点数
        return x
