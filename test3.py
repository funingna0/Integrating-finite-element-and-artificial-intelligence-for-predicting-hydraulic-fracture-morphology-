import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from modelClass import ImageGenerationCNN
# 定义模型（与你训练时相同的结构）


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageGenerationCNN().to(device)
model.load_state_dict(torch.load("model_epoch_1000.pth"))  # 加载训练好的权重
model.eval()  # 设置模型为评估模式


input_data={'时间/s':200,'排量/m3/min':3.5,'流体黏度':2,
            '隔层1/m':10,'隔层1x应力/Mpa':16,'隔层1y应力/Mpa':28,'隔层1z应力/Mpa':24,'隔层1杨氏模量/Gpa':39,'隔层1抗拉强度/Mpa':4,
            '储层1/m':20,'储层1x应力/Mpa':10,'储层1y应力/Mpa':25,'储层1z应力/Mpa':21,'储层1杨氏模量/Gpa':32,'储层1抗拉强度/Mpa':3,'储层1孔数':12,'储层1暂堵时刻/s':0,'储层1堵后孔数':12,
            '隔层2/m':10,'隔层2x应力/Mpa':16,'隔层2y应力/Mpa':28,'隔层2z应力/Mpa':24,'隔层2杨氏模量/Gpa':39,'隔层2抗拉强度/Mpa':4,
            '储层2/m':20,'储层2x应力/Mpa':10,'储层2y应力/Mpa':25,'储层2z应力/Mpa':21,'储层2杨氏模量/Gpa':32,'储层2抗拉强度/Mpa':3,'储层2孔数':12,'储层2暂堵时刻/s':0,'储层2堵后孔数':12,
            '隔层3/m':10,'隔层3x应力/Mpa':16,'隔层3y应力/Mpa':28,'隔层3z应力/Mpa':24,'隔层3杨氏模量/Gpa':39,'隔层3抗拉强度/Mpa':4,
            '储层3/m':20,'储层3x应力/Mpa':10,'储层3y应力/Mpa':25,'储层3z应力/Mpa':21,'储层3杨氏模量/Gpa':32,'储层3抗拉强度/Mpa':3,'储层3孔数':12,'储层3暂堵时刻/s':0,'储层3堵后孔数':12,
            '隔层4/m':10,'隔层4x应力/Mpa':16,'隔层4y应力/Mpa':28,'隔层4z应力/Mpa':24,'隔层4杨氏模量/Gpa':39,'隔层4抗拉强度/Mpa':4,

            }
input_data=list(input_data.values())
# 假设你的输入是54个参数
# input_data = [
#     200, 3.5, 2, 10, 16, 28, 24, 39, 4, 20, 10, 25, 21, 32, 3, 12, 0, 12, 10, 16, 28, 24, 39, 4, 20, 10, 25, 21, 32, 3, 12, 0,
#     12, 10, 16, 28, 24, 39, 4, 20, 10, 25, 21, 32, 3, 12, 0, 12, 10, 16, 28, 24, 39, 4
# ]

input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # 形状为[1, 54]

# 使用模型进行推断
with torch.no_grad():
    generated_image, generated_float = model(input_tensor)

# 处理生成的浮点数
float_value = generated_float.item()
float_str = str(float_value).replace('.', '_')

# 保存生成的图像
generated_image = generated_image.squeeze().cpu()  # 去除batch维度并将其移回CPU
generated_image = generated_image.permute(1, 2, 0)  # 转换为[H, W, C]格式

# 将生成的图像转换为PIL图像
generated_image = Image.fromarray((generated_image.numpy() * 255).astype(np.uint8))

# 保存到文件，文件名为浮点数值（替换小数点为下划线）
generated_image.save(f"generated_image_{float_str}.png")
print(f"Generated image saved as: generated_image_{float_str}.png")
