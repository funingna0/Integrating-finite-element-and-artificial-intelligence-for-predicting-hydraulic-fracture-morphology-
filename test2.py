import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from modelClass import ImageGenerationCNN1
# 定义模型（与你训练时相同的结构）
class ImageGenerationCNN(nn.Module):
    def __init__(self):
        super(ImageGenerationCNN, self).__init__()

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

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageGenerationCNN1().to(device)
model.load_state_dict(torch.load("best_model.pth"))  # 加载训练好的权重
model.eval()  # 设置模型为评估模式

# 假设你的输入是54个参数
# input_data = np.random.rand(54)  # 随机生成一些数据作为示例
input_data={'时间/s':100,'排量/m3/min':3.5,'流体黏度':2,
            '隔层1/m':10,'隔层1x应力/Mpa':16,'隔层1y应力/Mpa':28,'隔层1z应力/Mpa':24,'隔层1杨氏模量/Gpa':39,'隔层1抗拉强度/Mpa':4,
            '储层1/m':20,'储层1x应力/Mpa':10,'储层1y应力/Mpa':25,'储层1z应力/Mpa':21,'储层1杨氏模量/Gpa':24,'储层1抗拉强度/Mpa':3,'储层1孔数':12,'储层1暂堵时刻/s':0,'储层1堵后孔数':12,
            '隔层2/m':10,'隔层2x应力/Mpa':16,'隔层2y应力/Mpa':28,'隔层2z应力/Mpa':24,'隔层2杨氏模量/Gpa':39,'隔层2抗拉强度/Mpa':4,
            '储层2/m':20,'储层2x应力/Mpa':10,'储层2y应力/Mpa':25,'储层2z应力/Mpa':21,'储层2杨氏模量/Gpa':32,'储层2抗拉强度/Mpa':3,'储层2孔数':12,'储层2暂堵时刻/s':0,'储层2堵后孔数':12,
            '隔层3/m':10,'隔层3x应力/Mpa':16,'隔层3y应力/Mpa':28,'隔层3z应力/Mpa':24,'隔层3杨氏模量/Gpa':39,'隔层3抗拉强度/Mpa':4,
            '储层3/m':20,'储层3x应力/Mpa':10,'储层3y应力/Mpa':25,'储层3z应力/Mpa':21,'储层3杨氏模量/Gpa':24,'储层3抗拉强度/Mpa':3,'储层3孔数':12,'储层3暂堵时刻/s':0,'储层3堵后孔数':12,
            '隔层4/m':10,'隔层4x应力/Mpa':16,'隔层4y应力/Mpa':28,'隔层4z应力/Mpa':24,'隔层4杨氏模量/Gpa':39,'隔层4抗拉强度/Mpa':4,

            }
input_data=list(input_data.values())
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # 形状为[1, 54]

# 使用模型进行推断
with torch.no_grad():  # 不需要计算梯度
    generated_image = model(input_tensor)

# 生成的图像是一个Tensor，你可以将其转换为PIL图像以便显示或保存
generated_image = generated_image.squeeze().cpu()  # 去除batch维度并将其移回CPU
generated_image = generated_image.permute(1, 2, 0)  # 转换为[H, W, C]格式
# generated_image = (generated_image + 1) / 2  # 将[-1, 1]范围的值转换为[0, 1]

# 将生成的图像转换为PIL图像
generated_image = Image.fromarray((generated_image.numpy() * 255).astype(np.uint8))

# 显示生成的图像
# generated_image.show()

# 或者保存到文件
generated_image.save("a4000_51.png")

##################################################
#输出范围的调整
# input_data = list(input_data.values())
# input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # 形状为[1, 54]

# # 使用模型进行推断
# with torch.no_grad():  # 不需要计算梯度
#     generated_image = model(input_tensor)

# # 生成的图像是一个Tensor，去除batch维度并将其移回CPU
# generated_image = generated_image.squeeze().cpu()  # 去除batch维度，得到形状为[C, H, W]
# generated_image = generated_image.permute(1, 2, 0)  # 转换为[H, W, C]格式

# # 将[-1, 1]范围的值转换为[0, 1]
# generated_image = (generated_image + 1) / 2  # [0, 1] 范围

# # 将[0, 1]范围的值转换为[0, 255]并转换为整数
# generated_image = (generated_image * 255).numpy().astype(np.uint8)

# # 将生成的图像转换为PIL图像
# generated_image = Image.fromarray(generated_image)

# # 显示生成的图像
# # generated_image.show()

# # 或者保存到文件
# generated_image.save("4060t0.png")
