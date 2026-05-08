import numpy as np
import skimage as ski
from skimage import color as ski_color
from skimage import io as ski_io
from skimage import transform as ski_transform
from skimage import feature as ski_feature
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

img_xr_rgb = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/xiaoren.png"))
img_xr_gray = ski_color.rgb2gray(img_xr_rgb)                # 把 RGB 图转成灰度图
print("原图形状：", img_xr_rgb.shape)

img_xr_gray_min = ski_transform.resize(img_xr_gray, (100, 300))
print("缩小图像：", img_xr_gray_min.shape)

# NOTE: 根据特征，可以建分类模型，只用于机器学习模型，不会用于深度学习模型
# 提取 HOG 特征
hog_feature = ski_feature.hog(img_xr_gray_min)
print("HOG 特征：", hog_feature.shape)


# 提取 LBP 特征
img = np.uint8(img_xr_gray_min)
lbp_feature = ski_feature.local_binary_pattern(img, P=8, R=0.5).reshape(-1)
print("LBP 特征：", lbp_feature.shape)

# 提取 Haar 特征
corner_harris_feature = ski_feature.corner_harris(img).reshape(-1)
print("Haar 特征：", corner_harris_feature.shape)