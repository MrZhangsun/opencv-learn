"""
图像梯度（Image Gradient）本质上不是一个复杂的新东西，它就是一句话：
描述图像中“像素变化有多快、往哪个方向变化”

换句话说：
哪里变化大，哪里就是边缘。

两个重要量
1️⃣ 梯度方向分量
𝐺𝑥：水平方向变化（左右）
𝐺𝑦：垂直方向变化（上下）
2️⃣ 梯度幅值,表示变化有多剧烈（边缘强度）:
𝐺=np.sqrt(𝐺𝑥 ** 2 + 𝐺𝑦 ** 2)
3️⃣ 梯度方向, 表示边缘朝哪个方向
𝜃=arctan(𝐺𝑦 / 𝐺𝑥)

在计算机里怎么实现？
现实中不能真的“求导”，所以用梯度算子进行卷积近似导数。
梯度算子：
Sobel
Scharr
拉普拉斯
Canny
"""
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import numpy as np

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

img_small2 = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/small2.png"))
# img_small2 = cv.cvtColor(img_small2, cv.COLOR_BGR2GRAY)
# # cv.imshow("img_small2", img_small2)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
#
# blur = cv.GaussianBlur(img_small2, (5, 5), 0)
# print("原图中的一行：\n", img_small2[11], sep="")
# print("\n高斯平滑后图像中的一行：\n", blur[11], sep="")
#
# # 对比高斯平滑前后图像的局部变化
# # plt.plot(img_small2[11]) # 单条线，输入Y，X轴自动生成
# # plt.plot(blur[11])
# # plt.show()

"""
准备图像，在图像上绘制几个纹理，便于观察不同算子的作用
"""
img_xiao_ren = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/xiaoren.png"))
img_car = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/car.jpg"), 0)
img_xiao_ren = cv.cvtColor(img_xiao_ren, cv.COLOR_BGR2GRAY)
img_xiao_ren1 = np.copy(img_xiao_ren)
h, w = img_xiao_ren.shape
cv.line(img_xiao_ren1, (0, h // 3), (w, h // 3), (0, 255, 0), 5)
cv.line(img_xiao_ren1, (0, h * 2 // 3), (w, h * 2 // 3), (0, 255, 0), 5)
cv.line(img_xiao_ren1, (w // 3, 0), (w // 3, h), (0, 255, 0), 5)
cv.line(img_xiao_ren1, (w * 2 // 3, 0), (w * 2 // 3, h), (0, 255, 0), 5)
cv.line(img_xiao_ren1, (0, 0), (w, h), (0, 255, 0), 5)
cv.line(img_xiao_ren1, (0, h), (w, 0), (0, 255, 0), 1)
# cv.imshow("img_xiao_ren1", img_xiao_ren1)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Sobel 算子
cv.Sobel 是 OpenCV 中用于边缘检测的核心函数之一。它的核心原理是计算图像像素强度在 
水平 (x) 和 垂直 (y) 方向上的变化率（也就是导数或梯度）。图像中梯度变化剧烈的位置，通常就是物体的边缘。
Sobel 算子的一个主要优势是它结合了高斯平滑，因此相比其他简单算子，对图像中的噪声有一定的抑制能力，结果更鲁棒

 Sobel 算子是如何工作的？
简单来说，cv.Sobel 使用两个特殊的 3x3 卷积核（也可以更大）分别检测水平和垂直边缘：

水平方向 (检测垂直边缘)：检测左右方向的明暗变化。

text
Gx =  [-1, 0, +1]      Gy =  [-1, -2, -1]
      [-2, 0, +2]            [ 0,  0,  0]
      [-1, 0, +1]            [+1, +2, +1]
垂直方向 (检测水平边缘)：检测上下方向的明暗变化。

通过这两个核计算出每个点在 x 和 y 方向的梯度 Gx 和 Gy 后，再通过勾股定理或绝对值求和，
就能得到该点最终的边缘强度（梯度幅值）：G = sqrt(Gx^2 + Gy^2) 或 G = |Gx| + |Gy|。

src：	输入的原始图像。	通常需要是8位的灰度图 (uint8)。
ddepth：	输出图像的深度（数据类型）。	这是最关键的参数之一。由于梯度计算可能出现负值，直接用 -1 (与原图相同) 会导致负数被截断为0，从而丢失信息。推荐使用 cv.CV_64F，它可以保存负值，处理更精确。
dx：	x 方向导数的阶数。	计算 x 方向梯度设为 1，否则设为 0。
dy：	y 方向导数的阶数。	计算 y 方向梯度设为 1，否则设为 0。
ksize：Sobel 核的大小。	必须是 1, 3, 5 或 7。ksize=-1 时会使用更精确的 Scharr 算子 (3x3)，效果通常优于普通的 3x3 Sobel。
scale：计算导数值时的缩放因子。	默认为 1，即不缩放。
delta：在结果存储前加上的一个可选值。	默认为 0。
borderType：	像素外推法，处理图像边界。	通常使用默认值 cv.BORDER_DEFAULT。
"""
# # 一阶导数
# sobel_x = cv.Sobel(img_xiao_ren1, cv.CV_64F, 1, 0, ksize=5)
# sobel_y = cv.Sobel(img_xiao_ren1, cv.CV_64F, 0, 1, ksize=5)
# # 混合偏导 dx=1, dy=1：图像会显得很"稀疏"，只有对角线方向的纹理或角点才会响应，大部分平坦区域和纯水平/垂直边缘会被抑制
# sobel_xy = cv.Sobel(img_xiao_ren1, cv.CV_64F, 1, 1, ksize=5)
#
# # 二阶导数（即边缘的曲率变化）
# sobel2_x = cv.Sobel(img_xiao_ren1, cv.CV_64F, 2, 0, ksize=5)
# sobel2_y = cv.Sobel(img_xiao_ren1, cv.CV_64F, 0, 2, ksize=5)
# sobel2_xy = cv.Sobel(img_xiao_ren1, cv.CV_64F, 2, 2, ksize=5)
# sobel2_xy2 = np.sqrt(sobel2_x ** 2 + sobel2_y ** 2)
#
# sobel_xx = cv.Sobel(sobel_x, cv.CV_64F, 1, 0, ksize=5)
# sobel_yy = cv.Sobel(sobel_y, cv.CV_64F, 0, 1, ksize=5)
# sobel_xxy = cv.Sobel(sobel_x, cv.CV_64F, 0, 1, ksize=5)
# sobel_yyx = cv.Sobel(sobel_y, cv.CV_64F, 1, 0, ksize=5)
#
# fig = plt.figure("图像梯度",figsize=(20, 10))
# ax11 = fig.add_subplot(2, 7, 1, title="原图")
# ax11.imshow(img_xiao_ren1, cmap="gray")
# ax12 = fig.add_subplot(2, 7, 2, title="Sobel水平方向梯度")
# ax12.imshow(sobel_x, cmap="gray")
# ax13 = fig.add_subplot(2, 7, 3, title="Sobel垂直方向梯度")
# ax13.imshow(sobel_y, cmap="gray")
# ax14 = fig.add_subplot(2, 7, 4, title="Sobel 混合偏导")
# ax14.imshow(sobel_xy, cmap="gray")
# ax15 = fig.add_subplot(2, 7, 5, title="Sobel梯度幅值")
# # 如果你想要边缘检测效果，永远不要使用 dx=1, dy=1 的组合。正确的做法是分两次调用 cv.Sobel，然后手动计算梯度幅值。
# ax15.imshow(np.sqrt(sobel_x ** 2 + sobel_y ** 2), cmap="gray")
#
# ax16 = fig.add_subplot(2, 7, 6, title="Sobel二阶水平方向梯度")
# ax16.imshow(sobel2_x, cmap="gray")
# ax17 = fig.add_subplot(2, 7, 7, title="Sobel二阶垂直方向梯度")
# ax17.imshow(sobel2_y, cmap="gray")
# ax18 = fig.add_subplot(2, 7, 8, title="Sobel二阶混合偏导")
# ax18.imshow(sobel2_xy, cmap="gray")
# ax19 = fig.add_subplot(2, 7, 9, title="Sobel二阶梯度幅值")
# ax19.imshow(sobel2_xy2, cmap="gray")
# ax20 = fig.add_subplot(2, 7, 10, title="Sobel XX方向梯度")
# ax20.imshow(sobel_xx, cmap="gray")
# ax21 = fig.add_subplot(2, 7, 11, title="Sobel YY方向梯度")
# ax21.imshow(sobel_yy, cmap="gray")
# ax22 = fig.add_subplot(2, 7, 12, title="Sobel X1Y2方向梯度")
# ax22.imshow(sobel_xxy, cmap="gray")
# ax23 = fig.add_subplot(2, 7, 13, title="Sobel Y1X2方向梯度")
# ax23.imshow(sobel_yyx, cmap="gray")
#
# plt.show()

"""
自定义Sobel算子
水平高斯模糊，垂直计算梯度
kernel = np.asarray([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])
"""
# #%%
# # 自定义一个 kernel 核
# kernel = np.asarray([
#     [-1, -2, -1],
#     [0, 0, 0],
#     [1, 2, 1]
# ])
#
# # 做卷积操作
# # 第 2 个参数为：ddepth，输出图像深度
# sobely = cv.filter2D(img_xiao_ren1, cv.CV_64F, kernel)    # 提取水平的边缘信息
# sobelx = cv.filter2D(img_xiao_ren1, cv.CV_64F, kernel.T)  # 提取垂直的边缘信息
#
# # kernel等价于下面 kernel1 + kernel2
# kernel1 = np.asarray([[1, 2, 1]])  # 水平
# kernel2 = np.asarray([[-1],
#                       [0],
#                       [1]])        # 垂直
#
# # 提取垂直的边缘信息
# # 做卷积操作
# # sobelx = cv.filter2D(img, cv.CV_64F, kernel.T)      # 等价于下面 2 行
# img_blur = cv.filter2D(img_xiao_ren1, cv.CV_64F, kernel1.T)     # 垂直高斯平滑
# sobelx2 = cv.filter2D(img_blur, cv.CV_64F, kernel2.T)  # 水平梯度
#
# # 提取水平的边缘信息
# # sobely = cv.filter2D(img, cv.CV_64F, kernel)      # 等价于下面 2 行
# img_blur = cv.filter2D(img_xiao_ren1, cv.CV_64F, kernel1)     # 水平高斯平滑
# sobely2 = cv.filter2D(img_blur, cv.CV_64F, kernel2)  # 垂直梯度
#
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(231)
# plt.imshow(img_xiao_ren1, 'gray')
# plt.title('Original')
#
# plt.subplot(232)
# plt.imshow(sobelx, 'gray')  # 垂直
# plt.title('sobelx')
#
# plt.subplot(233)
# plt.imshow(sobely, 'gray')  # 水平
# plt.title('sobely')
#
# plt.subplot(235)
# plt.imshow(sobelx, 'gray')  # 垂直
# plt.title('sobelx2')
#
# plt.subplot(236)
# plt.imshow(sobely, 'gray')  # 水平
# plt.title('sobely2')
# plt.show()

"""
Scharr(沙尔)算子
是 Sobel 算子的改进版本，专门用于解决标准 Sobel 算子（3x3）在计算小规模梯度时的旋转不对称性问题。
数学原理对比
    Sobel 算子 (3x3)
    Gx =  [-1, 0, +1]      Gy =  [-1, -2, -1]
          [-2, 0, +2]            [ 0,  0,  0]
          [-1, 0, +1]            [+1, +2, +1]
    Scharr 算子 (3x3)
    Gx =  [-3, 0, +3]      Gy =  [-3, -10, -3]
          [-10, 0, +10]          [ 0,   0,  0]
          [-3, 0, +3]            [+3, +10, +3]
          
问题：Sobel 的旋转不对称性
标准 Sobel 算子对对角线方向的边缘检测精度低于水平/垂直方向，因为：
水平/垂直方向的权重：[1, 2, 1]
对角线方向的权重：[1, 0, 1]（实际效果较差）
这导致当图像旋转45度时，相同边缘的检测强度会变化约15%。

解决：Scharr 的最优性
Scharr 算子通过优化核权重，实现了更好的旋转对称性：
误差从 Sobel 的 ~15% 降低到 ~5%
对所有方向的边缘响应更一致
特别适合需要各向同性的梯度计算场景
"""

# scharr_x = cv.Scharr(img_xiao_ren1, cv.CV_64F, 1, 0) # 提取边缘信息（垂直）
# scharr_y = cv.Scharr(img_xiao_ren1, cv.CV_64F, 0, 1) # 提取边缘信息（水平）
# scharr_xxyy = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
#
# scharr_xy = cv.Scharr(scharr_x, cv.CV_64F, 0, 1)
# scharr_yx = cv.Scharr(scharr_y, cv.CV_64F, 1, 0)
# scharr_xyyx = np.sqrt(scharr_xy ** 2 + scharr_yx ** 2)
#
#
# plt.figure(figsize=(20, 10))
# plt.subplot(261)
# plt.imshow(img_xiao_ren1, 'gray')
# plt.title('Original')
# plt.subplot(262)
# plt.imshow(scharr_x, 'gray')
# plt.title('scharr_x')
# plt.subplot(263)
# plt.imshow(scharr_y, 'gray')
# plt.title('scharr_y')
# plt.subplot(265)
# plt.imshow(scharr_xy, 'gray')
# plt.title('scharr_xy')
# plt.subplot(266)
# plt.imshow(scharr_yx, 'gray')
# plt.title('scharr_yx')
# plt.subplot(267)
# plt.imshow(scharr_xyyx, 'gray')
# plt.title('scharr_xyyx')
# plt.subplot(268)
# plt.imshow(scharr_xxyy, 'gray')
# plt.title('scharr_xxyy')
#
# plt.show()

"""
自定义Scharr算子
"""
#
# kernel = np.asarray([
#     [-3, -10, -3],
#     [0, 0, 0],
#     [3, 10, 3]
# ])
# # 做卷积操作
# scharr_y = cv.filter2D(img_xiao_ren1, cv.CV_64F, kernel)    # 提取水平的边缘信息
# scharr_x = cv.filter2D(img_xiao_ren1, cv.CV_64F, kernel.T)  # 提取垂直的边缘信息
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(131)
# plt.imshow(img_xiao_ren1, 'gray')
# plt.title('Original')
#
# plt.subplot(132)
# plt.imshow(scharr_x, 'gray')  # 垂直
# plt.title('scharr_x')
#
# plt.subplot(133)
# plt.imshow(scharr_y, 'gray')  # 水平
# plt.title('scharr_y')
# plt.show()

"""
Laplacian 算子是一种二阶微分算子，用于检测图像中的边缘、角点和孤立点。
与 Sobel 和 Scharr 等一阶微分算子不同，Laplacian 对图像强度变化的变化率敏感，因此对噪声也更敏感。
数学原理
连续域定义
Laplacian 算子是梯度的散度，数学上表示为：
∇²I = ∂²I/∂x² + ∂²I/∂y²

数学本质：Laplacian 是一个零和算子
Laplacian 核的所有元素之和必须为 0，这是由它的数学定义决定的：

text
对于 4-邻域核：0 + (-1) + 0 + (-1) + 4 + (-1) + 0 + (-1) + 0 = 0
对于 8-邻域核：(-1)+(-1)+(-1)+(-1)+8+(-1)+(-1)+(-1)+(-1) = 0
为什么必须和为 0？
Laplacian 计算的是二阶导数，常数区域（平坦区域）的导数为 0

如果核的和不为 0，平坦区域也会产生非零响应，这就错了
离散化形式（3x3核）
    最常用的离散近似是(核中央数字的绝对值 = 周围数字之和，导致可以进行边缘检测)：
    L = [0,  -1,  0]    或   L = [-1, -1, -1]
        [-1,  4, -1]             [-1,  8, -1]
        [0,  -1,  0]             [-1, -1, -1]
        
    两种形式的区别：
    4邻域版本：只考虑上下左右（对角线不参与）
    8邻域版本：考虑所有8个方向（更常用）
"""
# ksize = 3
# laplacian_1 = cv.Laplacian(img_xiao_ren1, cv.CV_64F, ksize=ksize)
# laplacian = cv.convertScaleAbs(laplacian_1)
#
# sobel_x = cv.Sobel(img_xiao_ren1, cv.CV_64F, dx=1, dy=0, ksize=ksize)  # 提取边缘信息（垂直）
# sobel_x = cv.convertScaleAbs(sobel_x)
#
# sobel_y = cv.Sobel(img_xiao_ren1, cv.CV_64F, dx=0, dy=1, ksize=ksize)  # 提取边缘信息（水平）
# sobel_y = cv.convertScaleAbs(sobel_y)
#
# scharr_x = cv.Scharr(img_xiao_ren1, cv.CV_64F, dx=1, dy=0)             # 提取边缘信息（垂直）
# scharr_x = cv.convertScaleAbs(scharr_x)
#
# scharr_y = cv.Scharr(img_xiao_ren1, cv.CV_64F, dx=0, dy=1)             # 提取边缘信息（水平）
# scharr_y = cv.convertScaleAbs(scharr_y)
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(231)
# plt.imshow(img_xiao_ren1, 'gray')
# plt.title('img')
#
# plt.subplot(232)
# plt.imshow(sobel_x, 'gray')    # 垂直
# plt.title('sobel_x')
#
# plt.subplot(233)
# plt.imshow(sobel_y, 'gray')    # 水平
# plt.title('sobel_y')
#
# plt.subplot(234)
# plt.imshow(laplacian, 'gray')  # 所有方向
# plt.title('laplacian')
#
# plt.subplot(235)
# plt.imshow(scharr_x, 'gray')   # 垂直
# plt.title('scharr_x')
#
# plt.subplot(236)
# plt.imshow(scharr_y, 'gray')   # 水平
# plt.title('scharr_y')
# plt.show()

"""
在 Sobel 检测中，depth 对于结果的影响：<br/>
当输出的 depth 设置为比较低的数据格式，那么当梯度值计算为负值的时候，就会将其重置为 0，从而导致失真。

在 Laplacian 检测中，该问题不大。
"""
# # 构建一个黑色图像
# img = np.zeros((300, 300), np.uint8)
#
# # 在黑底上加白框
# img[100:200, 100:200] = 255

# 构建白底黑框的图像
# img = np.ones((300, 300), np.uint8) * 255
# img[100:200, 100:200] = 0
#
# ksize = 5
# dst1 = cv.Laplacian(img, cv.CV_8U, ksize=ksize)   # 数据格式低
# dst2 = cv.Laplacian(img, cv.CV_64F, ksize=ksize)  # 数据格式高
#
# # 求绝对值 转成 8 位无符号灰度图
# dst3 = np.uint8(np.absolute(dst2)) # 数据格式低
# print(img[0, 0])
# print(dst1[0, 0])
# print(dst2[0, 0])
# print(dst3[0, 0])
#
#
# # 做 Sobel 的操作
# dst4 = cv.Sobel(img, cv.CV_8U, dx=1, dy=0, ksize=ksize)   # 数据格式低
# dst5 = cv.Sobel(img, cv.CV_64F, dx=1, dy=0, ksize=ksize)  # 数据格式高
# dst6 = np.uint8(np.absolute(dst5))                        # 数据格式低
#
# # c. 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(241)
# plt.imshow(img, 'gray')
# plt.title('img')
#
# plt.subplot(242)
# plt.imshow(dst1, 'gray')  # 低：1 条线
# plt.title('Laplacian1')
#
# plt.subplot(243)
# plt.imshow(dst2, 'gray')  # 高：2 条线（外边白线，里边黑线）
# plt.title('Laplacian2')
#
# plt.subplot(244)
# plt.imshow(dst3, 'gray')  # 低：2 条线（求绝对值后，里边黑线变成白线）
# plt.title('Laplacian3')
#
# plt.subplot(246)
# plt.imshow(dst4, 'gray')  # 低：1 条线（不支持负数，只显示左边 255-0=255 的梯度，不显示右边 0-255=-255 的梯度）
# plt.title('Sobel1')
#
# plt.subplot(247)
# plt.imshow(dst5, 'gray')  # 高：2 条线（浮点数：左边很大的正数，右边很小的负数，灰色部分都为 0；
#                                      # 可视化：很大的正数映射为 255 白色，很小的负数映射为 0 黑色，原有的 0 区域映射为中间值灰色）
# plt.title('Sobel2')
#
# plt.subplot(248)
# plt.imshow(dst6, 'gray')  # 低：2 条线（求绝对值后，里边黑线变成白线，原有的 0 区域仍然为 0）
# plt.title('Sobel3')
#
# plt.show()

"""
车牌号轮廓检查
"""
# img_car1 = cv.GaussianBlur(img_car, (5, 5), 2)  # 高斯平滑（2：标准差，控制模糊强度）
# img_car2 = cv.Laplacian(img_car1, cv.CV_64F, ksize=5)
# img_car3 = np.uint8(np.absolute(img_car2))
#
# # 二值化处理
# ret, img_car4 = cv.threshold(img_car3, 127, 255, cv.THRESH_BINARY)
#
# # 膨胀操作（水平）
# iterations=4 #：操作次数
# kernel = np.asarray([ [1, 1, 1, 1, 1, 1, 1, 1]])
# img_car5 = cv.morphologyEx(img_car4, cv.MORPH_DILATE, kernel, iterations=iterations)
#
#
# cv.imshow('Original Image', img_car)
# cv.imshow('Gaussian Smoothing', img_car1)
# cv.imshow('Laplacian Edge Extraction', img_car2)
# cv.imshow('Convert to Uint8 Integer', img_car3)
# cv.imshow('Binarization', img_car4)
# cv.imshow('Dilation Operation (Horizontal)', img_car5)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
Canny

像素类型	判断条件	处理结果
强边缘	梯度 > 200	✅ 直接保留（确定是边缘）
弱边缘	100 < 梯度 < 200	🤔 有条件保留（仅当与强边缘相连时才保留）
非边缘	梯度 < 100	❌ 直接丢弃（确定不是边缘）
"""
# # a. 高斯去噪
# # sigmX = 0：会根据核大小自动计算标准差，表示高斯强度
# blur = cv.GaussianBlur(img_xiao_ren1, (5, 5), 0)
#
# # b. Canny边缘检测
# # threshold1=50：低阈值，梯度值低于此阈值的边缘被丢弃
# # threshold2=250：高阈值，梯度值高于此阈值的边缘被保留为强边缘
# # threshold1 < 梯度 < threshold2: 边缘与强边缘相连的时候保留，否则丢弃
# edges = cv.Canny(blur, 20, 80)
#
# laplacian = cv.Laplacian(blur, cv.CV_64F, ksize=5)
# laplacian = cv.convertScaleAbs(laplacian)
#
# # 可视化
# plt.figure(figsize=(10, 10))
#
# plt.subplot(221)
# plt.imshow(img_xiao_ren1, cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(blur, cmap='gray')       # 高斯去噪
# plt.title('Gaussian Blur Image')
#
# plt.subplot(223)
# plt.imshow(edges, cmap='gray')      # Canny 边缘检测
# plt.title('Canny Edge Image')
#
# plt.subplot(224)
# plt.imshow(laplacian, cmap='gray')  # laplacian 边缘检测
# plt.title('laplacian Edge Image')
# # plt.show()
#
# # 将灰度图转成 BGR 格式
# bgr_xiaoren = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
# # 画矩形
# cv.rectangle(bgr_xiaoren, pt1=(243, 243), pt2=(253, 253), color=(0, 0, 255), thickness=2)
# # 可视化
# cv.imshow("image", bgr_xiaoren)
#
# cv.waitKey(0)
# # 释放指定窗口资源
# cv.destroyWindow('image')

# 高斯去噪，标准差为 2
blur = cv.GaussianBlur(img_car, (5, 5), 2)

# 提取图像边缘（二值化图）
# threshold1=50：低阈值，梯度值低于此阈值的边缘被丢弃
# threshold2=250：高阈值，梯度值高于此阈值的边缘被保留为强边缘（输出为白色）
edges = cv.Canny(blur, threshold1=50, threshold2=250)

kernel = np.asarray([
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])

morph_dilate = cv.morphologyEx(edges, cv.MORPH_DILATE, kernel, iterations=2)

cv.imshow('1.Original Image', img_car)
cv.imshow('2.Gaussian Blur', blur)
cv.imshow('3.Canny Edge Image', edges)
cv.imshow('4.MORPH DILATE', morph_dilate)

cv.waitKey(0)
cv.destroyAllWindows()
