"""
图像二值化
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pathlib import Path

from torchvision.transforms.v2 import GaussianBlur

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

# 一条横线
# .arange(255, -1, -1)：stop=-1，step=-1
# .reshape(1, -1)：-1表示会自动计算数值
area = np.arange(255, -1, -1, dtype=np.uint8).reshape(1, -1)

for i in range(area.shape[1]):
    arr_i = np.arange(255, -1, -1, dtype=np.uint8).reshape(1, -1)
    area = np.append(area, arr_i, axis=0)
    print(i)

# 普通二值化操作，将小于等于阈值 thresh 的设置为 0，大于该值的设置为 maxval（常用）
# ret：实际使用的阈值
# thresh1：二值化后的输出图像
thresh1, area1 = cv.threshold(area, 127, 255, cv.THRESH_BINARY)
# 反转的二值化操作，将小于等于阈值 thresh 的设置为 maxval，大于该值的设置为 0
thresh2, area2 = cv.threshold(area, 127, 255, cv.THRESH_BINARY_INV)
thresh3, area3 = cv.threshold(area, 127, 255, cv.THRESH_TRUNC)
# 0 二值化操作，将小于等于阈值的设置为 0，大于该值的设置为原始值
thresh4, area4 = cv.threshold(area, 127, 255, cv.THRESH_TOZERO)
# 反转 0 二值化操作，将小于等于阈值的设置为原始值，大于阈值的设置为 0
thresh5, area5 = cv.threshold(area, 127, 255, cv.THRESH_TOZERO_INV)

# 显示
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [area, area1, area2, area3, area4, area5]

fig1 = plt.figure("图像1")
for index in range(len(images)):
    fig1.add_subplot(2, 3, index + 1)
    plt.imshow(images[index], 'gray')
    plt.title(titles[index])
    plt.xticks([])
    plt.yticks([])

# plt.show()

""""
自适应二值化操作: 这是OpenCV中的自适应阈值函数，能够根据图像局部区域的像素值自动计算阈值，
比全局阈值更适合处理光照不均匀的图像。

cv.adaptiveThreshold(
    src,                    # 输入图像（必须是灰度图，8-bit单通道）
    maxValue,               # 满足条件时赋予的最大值（通常255）
    adaptiveMethod,         # 自适应方法：MEAN_C 或 GAUSSIAN_C
    thresholdType,          # 阈值类型：THRESH_BINARY 或 THRESH_BINARY_INV
    blockSize,              # 邻域大小（奇数，如3,5,7,11...）
    C                       # 从均值/加权均值中减去的常数
) -> dst                    # 输出二值图像
    
# 使用均值的方式产生当前像素点对应的阈值
# 对每个像素 (x,y)，计算其 blockSize×blockSize 邻域内的平均值减去 C 的值
# 比周围颜色深的颜色会变成黑色：像素点颜色值 - (均值 - C) ≤ 0 -> 黑色
# 比周围颜色浅的颜色会变成白色：像素点颜色值 - (均值 - C) > 0 -> 白色
# C=2：C大于 0，阈值(均值 - C)变小，黑色区域变小，白色区域变大
# 突出边缘信息
"""
img1 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png",
                 cv.IMREAD_GRAYSCALE)
# 普通二值化
ret, bin0 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)

# 自适应均值二值化
bin1 = cv.adaptiveThreshold(img1, 255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)

# 自适应高斯二值化
bin2 = cv.adaptiveThreshold(img1, 255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)


# 图像效果展示
titles = ['原图', '普通二值化', '自适应-均值', '自适应-高斯']
images = [img1, bin0, bin1, bin2]
fig2 = plt.figure("图像2", figsize=(6, 7))
fig2.suptitle("不同二值化对比")
for index in range(len(images)):
    ax = fig2.add_subplot(2, 2, index + 1)
    ax.set_title(titles[index])
    ax.imshow(images[index], 'gray')
    ax.set_xticks([])
    ax.set_yticks([])

plt.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.1,
                    wspace=0.2, hspace=0.2)

plt.close(fig1)
plt.close(fig2)

"""
正太分布曲线的
✅ 对称轴：曲线关于直线 
x=μ 左右对称
✅ 峰值位置：曲线最高点位于 
x=μ 处
✅ 中心位置：分布的中心（重心）在 
x=μ 处
✅ 中位数 = 众数 = 均值：三者重合于同一点
"""
# 产生噪音数据
# 适合严格范围控制（从均匀分布中抽取的随机样本）
# img1 = np.random.uniform(low=0, high=255, size=(300, 300))

# 返回从正态分布中抽取的随机样本
# 适合真实数据模拟：均值 150，标准差 100
img1 = np.random.normal(150, 100, size=(300, 300))
# 裁剪 把 img1 中所有像素值限制在 0 ~ 255 之间
img1 = np.clip(img1, 0, 255)
img1 = np.astype(img1, np.uint8)

img2 = np.zeros((300, 300), dtype=np.uint8) # 黑色背景
# print(img2.shape)
img2[100:200, 100:200] = 255 # 中间部分白色

# dst=src1⋅α+src2⋅β+γ
# alpha + beta = 1（亮度自然）
img_merge = cv.addWeighted(src1=img1, alpha=0.3, src2=img2, beta=0.3, gamma=0)

# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img_merge", img_merge)
# cv.waitKey(0)
# cv.destroyAllWindows()

# plt.imshow(img_merge, 'gray')
# plt.axis('off')
# plt.show()

# 进行普通二值化操作
# ret：实际使用的阈值
# th1：二值化后的输出图像
th1, ret1 = cv.threshold(img_merge, 127, 255, cv.THRESH_BINARY)
th2, ret2 = cv.threshold(img_merge, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 找到一个阈值，让“前景”和“背景”的差异最大
blur = cv.GaussianBlur(src=img_merge, ksize=(5, 5), sigmaX=0)
th3, ret3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

images = [img_merge, 0, ret1,
          img_merge, 0, ret2,
          blur, 0, ret3]

titles = ["Original Noisy Image", "Histogram", "Global Thresholding(v=127)",
          "Original Noisy Image", "Histogram", f"Otsu's Thresholding(v={ret2})",
          "Gaussian filtered Image", "Histogram", f"Otsu's Thresholding(v={ret3})"]

fig = plt.figure("二值化")
fig.suptitle("二值化效果对比")

for i in range(3):
    # 原始图
    ax = fig.add_subplot(3, 3, i * 3 + 1)
    ax.set_title(titles[i * 3])
    ax.imshow(images[i * 3], 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # 直方图
    ax1 = fig.add_subplot(3, 3, i * 3 + 2)
    ax1.set_title(titles[i * 3 + 1])
    # 绘制图像的灰度直方图，用于分析图像的像素值分布
    # .ravel()：将多维数组（图像矩阵）展平为一维数组，以便统计像素值
    # 256：直方图的区间数量，表示将像素值范围 [0,255] 分成 256 个区间统计频数，每个区间对应一个灰度级
    ax1.hist(images[i * 3].ravel(), 256) # hist() 需要的是“一维数据”
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 二值化后的图
    ax2 = fig.add_subplot(3, 3, i * 3 + 3)
    ax2.set_title(titles[i * 3 + 2])
    ax2.imshow(images[i * 3 + 2], 'gray')
    ax2.set_xticks([])
    ax2.set_yticks([])

# plt.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.1,)
# plt.show()




