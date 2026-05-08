# 色彩空间转换
import cv2 as cv
import numpy as np
from pathlib import Path
"""
颜色检测 → 用 HSV
边缘/结构 → 用 Gray
深度学习 → 用 RGB
光照鲁棒 → 用 LAB

| 色彩空间 | 含义       |
| ---- | -------- |
| BGR  | OpenCV默认 |
| RGB  | 人类视觉     |
| HSV  | 颜色分离     |
| LAB  | 感知均匀     |
| Gray | 单通道      |

openCV色彩空间转换API，cv.cvtColor
"""
img_path = Path(__file__).parent.parent.parent.parent / "assets/car.jpg"
img = cv.imread(img_path, flags=1) # 默认flags = 1, BGR; flags=0,加载灰度图
print(type(img))
print(img.shape)
cv.imshow('BGR', img)
cv.waitKey(-1)
cv.destroyWindow('BGR') # 释放命名窗口

"""
RGB（更符合人类认知）
用于 matplotlib 显示（否则颜色会错）
通道顺序：R, G, B
| 属性 | 说明      |
| -- | ------- |
| 结构 | 3通道     |
| 范围 | 0 ~ 255 |
| 含义 | 每个通道强度  |
图像显示
深度学习输入
基础处理
"""
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print(type(img_rgb))
print(img_rgb.shape)
cv.imshow('RGB', img_rgb)
cv.waitKey(-1)

"""
灰度图（Gray）单通道，常用于：
边缘检测（如 Canny）
特征提取
机器学习输入
是一个二维数组，数组的值代表intensity（亮度）是根据RGB图像加权计算得到的，公式如下：
Gray=0.299R+0.587G+0.114B

| 属性 | 说明      |
| -- | ------- |
| 结构 | 单通道     |
| 范围 | 0 ~ 255 |
| 含义 | 亮度      |

"""
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(type(img_gray))
print(img_gray.shape)
cv.imshow('GRAY', img_gray)
cv.waitKey(-1)

"""
HSV = 色相(H) + 饱和度(S) + 明度(V)

| 通道 | 含义  | OpenCV范围   |
| -- | --- | ---------- |
| H  | 色相  | 0 ~ 179 ⚠️ |
| S  | 饱和度 | 0 ~ 255    |
| V  | 明度  | 0 ~ 255    |
色相H，标准范围是0～360，open CV里面是0～179
使用场景（非常重要）：
颜色识别（红/绿/蓝）
目标分割
抗光照变化

"""
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
print(type(img_hsv))
print(img_hsv.shape)
print(img_hsv[1, 1, :])
cv.imshow('HSV', img_hsv)
cv.waitKey(-1)

"""
LAB（高级视觉）
| 通道 | 含义    | 范围              |
| -- | ----- | --------------- |
| L  | 亮度    | 0 ~ 255         |
| A  | 绿 ↔ 红 | 0 ~ 255（128为中性） |
| B  | 蓝 ↔ 黄 | 0 ~ 255（128为中性） |
👉 使用：
图像增强
风格迁移
光照鲁棒处理
"""
img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
print(type(img_lab))
print(img_lab.shape)
cv.imshow('LAB', img_lab)
cv.waitKey(-1)

"""
YCrCb（视频/人脸）
| 通道 | 含义  | 范围      |
| -- | --- | ------- |
| Y  | 亮度  | 0 ~ 255 |
| Cr | 红色差 | 0 ~ 255 |
| Cb | 蓝色差 | 0 ~ 255 |
使用：

人脸检测（肤色分布明显）
视频压缩（如 H.264）
"""
img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
print(type(img_ycrcb))
print(img_ycrcb.shape)
cv.imshow('YCRCB', img_ycrcb)
cv.waitKey(-1)
cv.destroyAllWindows()

# 灰度图片怎么是一个二维数组？色彩空间转换后，为什么图像的视觉显示效果发生了变化？