import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path

cwd = Path(__file__).resolve().parent.parent
# 读取图片
img = cv2.imread(f"{cwd}/assets/car.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转 tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
x = transform(img).unsqueeze(0)

# 模型
model = models.resnet18(pretrained=True)

# 取第一层输出
with torch.no_grad():
    feature = model.conv1(x)

# 可视化前16个特征图
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(feature[0, i].numpy(), cmap='gray')
    ax.axis('off')

plt.show()