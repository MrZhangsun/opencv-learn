from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt

"""
图像平滑（Image Smoothing）本质上就是一句话：
让图像“变得更柔和”，减少噪声和细节波动
用邻域像素“平均”当前像素，从而减少噪声，让图像更平滑（但更模糊）

图像平滑其实是在做：
用周围像素的值，来“重新计算当前像素”
举个简单例子：
原来某个像素：100
周围像素：90, 100, 110
👉 平滑后：≈ (90 + 100 + 110) / 3 = 100
👉 如果有噪声（比如突然出现 255），就会被“平均掉”

为什么要做图像平滑？
非常重要的一步，通常用于
1️⃣ 去噪（最常见）
    摄像头噪声
    压缩噪声
2️⃣ 为后续处理做准备
    比如你前面做的：
    cv.threshold（二值化）
    Otsu
    边缘检测
    👉 不平滑会导致：
        边缘乱
        阈值不准
3️⃣ 提高鲁棒性
让算法对“细小波动不敏感”


常见的4种平滑方法
1️⃣ 均值滤波（最简单）
cv.blur(img, (5, 5))
每个像素 = 周围平均值
2️⃣ 高斯滤波（最常用 ⭐）
cv.GaussianBlur(img, (5, 5), 0)
越近的像素权重越高（符合正态分布）
✔️ 去噪效果好
✔️ 保留一定边缘
3️⃣ 中值滤波（去椒盐噪声神器）
cv.medianBlur(img, 5)
👉 取“中间值”，不是平均
✔️ 去除黑白点（salt & pepper noise）
✔️ 保边缘
4️⃣ 双边滤波（高级）
cv.bilateralFilter(img, 9, 75, 75)
👉 同时考虑：
    空间距离
    像素差异
    ✔️ 保边缘最强
    ❌ 计算慢
"""

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

# 加载图像
img1 = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/xiaoren.png"))
koala = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/koala.png"))

"""
自定义卷积操作
"""
# # 自定义卷积操作:自定义一个 kernel 核
# kernel = np.ones((3, 3), dtype=np.float32) / 9 # 这里除9表示归一化，避免像素值变大或溢出
#
# # 做卷积操作
# # 第 2 个参数为：ddepth，输出图像深度（-1 表示与输入相同）
# # 图像深度指单个像素值的存储格式，例如 cv.CV_8U（8 位无符号）
# img2 = cv.filter2D(img1, -1, kernel)
#
# # 缩小 1/2
# h, w, _ = img1.shape
# h_new = int(h / 2)
# w_new = int(w / 2)
# img3 = cv.resize(img1, (w_new, h_new))
# img4 = cv.resize(img2, (w_new, h_new))
#
#
# # 放大 1/5
# img5 = cv.resize(img3, (int(w_new * 0.2), int(h_new * 0.2))) # 缩小
# img6 = cv.resize(cv.filter2D(img4, -1, kernel), (int(w_new * 0.2), int(h_new * 0.2))) # 卷积并缩小
#
# # 可视化
# plt.subplot(231)
# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))  # 显示转换后的 RGB 图像
#
# plt.subplot(232)
# plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))  # 缩小 1 次
#
# plt.subplot(233)
# plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))  # 缩小 2 次
#
# plt.subplot(234)
# plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))  # 卷积 1 次
#
# plt.subplot(235)
# plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))  # 卷积 1 次 -> 缩小 1 次
#
# plt.subplot(236)
# plt.imshow(cv.cvtColor(img6, cv.COLOR_BGR2RGB))  # 卷积并缩小 2 次
#
# plt.show()

"""
均值滤波
dst = cv.blur(img, ksize=(11, 11))
"""

# 做卷积操作
# 核越大，越模糊
# dst = cv.blur(img1, ksize=(11, 11)) # 均值滤波
# # dst = cv.blur(img1, ksize=(3, 3)) # 均值滤波
#
# # kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25
# # dst = cv.filter2D(img1, -1, kernel)
#
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
# plt.title('原图')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
# plt.title('均值滤波')
# plt.show()

"""
高斯模糊: 用“高斯分布权重”对图像做加权平均，让图像变平滑，更直观一点

普通平均（均值滤波）是：
每个像素 = 周围像素的平均值（权重都一样）

而高斯模糊是：
每个像素 = 周围像素的加权平均（越近权重越大）

高斯卷积核长什么样？
典型 5×5：

1   4   7   4   1
4  16  26  16   4
7  26  41  26   7
4  16  26  16   4
1   4   7   4   1
中心最大，向外递减

ksize=(5, 5)：高斯核大小
sigmaX = 2.0
👉 控制“横向模糊程度”
可以理解为：
模糊影响范围
小 → 模糊弱
大 → 模糊强
sigmaY = 2.0
👉 纵向模糊程度
"""

# img2 = cv.GaussianBlur(img1, (5, 5), 2.0, sigmaY=2.0)
# kernel = cv.getGaussianKernel(ksize=5, sigma=1, ktype=cv.CV_32F)
# print("\n一维高斯核：\n", kernel, sep="")
#
# """
# kernel = np.dot(a, a.T)
# 结果：
#
# [1*1   1*2   1*1]   → [1 2 1]
# [2*1   2*2   2*1]   → [2 4 2]
# [1*1   1*2   1*1]   → [1 2 1]
# """
# kernel = np.dot(kernel, kernel.T)
# print("\n二维高斯核：\n", kernel, sep="")
#
# print("\n原图：\n", img1.shape, sep="")
# print("\n高斯模糊后：\n", img2.shape, sep="")
# fig = plt.figure(figsize=(20, 10))
# ax1 = fig.add_subplot(2, 6, 1)
# ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB)) # 原图
# ax1.set_title("原图")
#
# ax2 = fig.add_subplot(2, 6, 7)
# ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
# ax2.set_title("高斯模糊")
#
# # 循环缩小图像
# for i in range(5):
#     h, w, _ = img1.shape  # 高度、宽度
#     w = int(w * 0.5)
#     h = int(h * 0.5)
#
#     img1 = cv.resize(img1, (w, h))  # 缩小图像
#     ax3 = fig.add_subplot(2, 6, i + 2)
#     ax3.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
#     ax3.set_title("原图缩小" + str(i + 1) + "次")
#
#     g_img = cv.GaussianBlur(img1, ksize=(5, 5), sigmaX=2.0, sigmaY=2.0)  # 高斯模糊
#     img2 = cv.resize(g_img, (w, h))  # 高斯模糊 + 缩小图像
#     ax4 = fig.add_subplot(2, 6, i + 8)
#     ax4.set_title("高斯模糊缩小" + str(i + 1) + "次")
#     ax4.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
# plt.show()

""""
自定义高斯滤波器
"""
# kernel_v = cv.getGaussianKernel(ksize=9, sigma=2.0, ktype=cv.CV_32F)
# kernel_h = np.transpose(kernel_v)
# print("垂直高斯核：\n", kernel_v, sep="")
# print("\n水平高斯核：\n", kernel_h, sep="")
#
# koala_v = cv.filter2D(koala, -1, kernel_v) # 垂直卷积
# koala_h = cv.filter2D(koala_v, -1, kernel_h) # 水平卷积
#
# # koala_v = np.clip(koala_v, 0, 255)
# # koala_h = np.clip(koala_h, 0, 255)
#
# fig = plt.figure(figsize=(20, 10))
# ax_v = fig.add_subplot(1, 3, 1)
# ax_h = fig.add_subplot(1, 3, 2)
# ax = fig.add_subplot(1, 3, 3)
#
# ax_v.imshow(cv.cvtColor(koala_v, cv.COLOR_BGR2RGB))
# ax_h.imshow(cv.cvtColor(koala_h, cv.COLOR_BGR2RGB))
# ax.imshow(cv.cvtColor(koala, cv.COLOR_BGR2RGB))
#
# ax_v.set_title("高斯垂直卷积")
# ax_h.set_title("高斯垂直/水平卷积")
# ax.set_title("原图")
#
# plt.show()

"""
自定义卷积核
"""
# 用一个“中心强抑制 + 周围加权”的卷积核，提取图像的边缘信息
# kernel = np.array([
#     [1, 2, 1],
#     [0, -8, 0],
#     [1, 2, 1]
# ], dtype=np.float32)

# 纯边缘检测（更干净）
# kernel = np.array([
#     [-1,-1,-1],
#     [-1, 8,-1],
#     [-1,-1,-1]
# ])

# # 锐化（边缘 + 原图）
# kernel = np.array([
#     [0,-1,0],
#     [-1,5,-1],
#     [0,-1,0]
# ])
# # 做一个卷积操作
# # 第 2 个参数为：ddepth，输出图像深度（-1 表示与输入相同）
# dst = cv.filter2D(img1, -1, kernel)
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(121)
# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
# plt.title('Original')
#
# plt.subplot(122)
# plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))  # 自定义卷积核
# plt.title('Define Kernel')
# plt.show()

"""
中值滤波
选择窗口区域中的中值作为输出值
dst = cv.medianBlur(img, ksize=5)
"""
#
# # 加噪声
# # 从正态分布中抽取的随机样本，均值和标准差都为 10
# noisy_img = np.random.normal(10, 10, (img1.shape[0], img1.shape[1], img1.shape[2]))
# # 截取 0-255 的值
# noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
# # 合成原图和噪声图（由于是 uint8 格式，所以如果出现求和大于 255 的情况，也会自动缩小到 255 以内）
#
# # img_merge = cv.addWeighted(src1=img1, alpha=0.3, src2=img2, beta=0.3, gamma=0)
# # img_merge = cv.add(img1, noisy_img)
# img_merge = img1 + noisy_img
# dst = cv.medianBlur(cv.cvtColor(img_merge, cv.COLOR_BGR2GRAY), ksize=5)
#
# fig = plt.figure(figsize=(20, 10))
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.set_title("原图")
# ax1.imshow(img_merge, 'gray')
#
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.set_title("中值滤波")
# ax2.imshow(dst, 'gray')
#
# plt.show()

"""
双边滤波: 提取图像的纹理、条纹信息
dst = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
# 做双边滤波
# d=9：滤波时考虑的邻域直径
# sigmaColor=75：颜色空间的标准差，值越大，颜色差异容忍度越高
# sigmaSpace=75：坐标空间的标准差，值越大，远处像素影响越大
"""
img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
dst = cv.bilateralFilter(img2, d=9, sigmaColor=75, sigmaSpace=75)
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("原图")
ax1.imshow(img2, 'gray')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("双边滤波")
ax2.imshow(dst, 'gray')
plt.show()
