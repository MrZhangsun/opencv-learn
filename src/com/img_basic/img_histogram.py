"""
什么是图像直方图（Histogram）
1️⃣ 本质
图像直方图就是：
👉 统计每个像素值出现的次数

如果是灰度图：
像素取值范围：0 ~ 255
横轴：灰度值（0~255）
纵轴：该灰度出现的频率

可以理解为：
“这张图是偏暗还是偏亮？对比度高不高？”

2️⃣ 举个例子
假设一张图：
[0, 0, 0, 128, 128, 255]
直方图就是：
| 灰度值 | 数量 |
| --- | -- |
| 0   | 3  |
| 128 | 2  |
| 255 | 1  |
3️⃣ 直方图能看出什么？
✅ 亮度
    集中在左 → 偏暗
    集中在右 → 偏亮
✅ 对比度
    分布很窄 → 对比度低（灰蒙蒙）
    分布很广 → 对比度高
✅ 图像质量
    是否过曝 / 欠曝

4️⃣ 常见操作
（1）直方图均衡化
    👉 提升对比度
    核心思想：把“拥挤的灰度”拉开
（2）直方图匹配（Histogram Matching）
    👉 把一张图变得“像另一张图”
    比如：
    把夜景图变成白天风格
    医学图像标准化

直方图的应用：
1. 图像检索
    直方图能用于图像检索，是因为：
    它把图像转成“分布特征向量”，可以用数学距离衡量相似度，同时对位置变化具有天然鲁棒性。
2. 图像特征向量

"""

from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

img_xiaoren_rgb = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/xiaoren.png"))
img_xiaoren_gray = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/xiaoren.png"), cv.IMREAD_GRAYSCALE)
img_koala = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/koala.png"), 0)
img_koala_rgb = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/koala.png"),)
img_t_rgb = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/t.png"),)
img_tsukuba_rgb = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/tsukuba.png"),)
img_template_gray = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/template.png"), cv.IMREAD_GRAYSCALE)

"""
常见的4种直方图的统计方式：
1. 基于Open CV的统计：
# channels=[0]：指定要计算直方图的通道索引（灰度图只有 1 个通道）
# mask=None：不使用掩膜
# histSize=[256]：将灰度级分成 256 个柱子（bin），每个柱子对应一个灰度值，范围 0-255
# ranges=[0, 256]：统计灰度值从 0 到 255（256 是上限，不包含）
# hist1：直方图数据，是一个形状为(256, 1)的二维数组，每个元素对应一个灰度级的像素数量
2. 基于Numpy的统计：
# img.ravel()：输入图像的一维扁平化
# 256：直方图的柱子数量
# [0, 256]：统计的像素值范围
# hist2：直方图数据，是一个长度为 256 的一维数组，每个元素表示对应灰度级的像素数量
3.基于Matplotlib的统计
ax5.hist(img_koala.ravel(), bins=256, range=(0, 256))
"""
# hist1 = cv.calcHist([img_koala], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
#
# hist2, bins = np.histogram(img_koala.ravel(), bins=256, range=(0, 256))
# # 和 np.histogram 一样的计算方式，但是效率快 10 倍
# hist3 = np.bincount(img_koala.ravel(), minlength=256)
#
#
# fig = plt.figure("直方图")
# ax1 = fig.add_subplot(231)
# ax1.imshow(img_koala, 'gray')
# ax1.set_title("原图")
#
# ax2 = fig.add_subplot(232)
# ax2.plot(hist1)
# ax2.set_title("CV直方图")
#
# ax3 = fig.add_subplot(233)
# ax3.plot(hist2)
# ax3.set_title("Numpy直方图")
#
# ax4 = fig.add_subplot(234)
# ax4.plot(hist3)
# ax4.set_title("Numpy直方图2")
#
# ax5 = fig.add_subplot(235)
# ax5.hist(img_koala.ravel(), bins=256, range=(0, 256))
# ax5.set_title("Matplotlib直方图")
#
# plt.tight_layout()
# plt.show()


"""
统计彩色图的直方图
BGR色彩通道：蓝，绿，红
HSV色彩通道：三条线，分别表示色调，明度，饱和度
"""
# fig = plt.figure("彩色直方图", figsize=(12, 8))

# # 使用枚举获取行索引
# for row_idx, w in enumerate(('CV', 'Numpy-01', 'Numpy-02', 'Matplotlib')):
#     for i, c in enumerate(("b", "g", "r")):
#         # 全局索引 = 行索引 × 每行子图数 + 列索引 + 1
#         global_idx = row_idx * 3 + i + 1
#         ax = fig.add_subplot(4, 3, global_idx)
#
#         if w == 'CV':
#             cv_hist = cv.calcHist([img_koala_rgb], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
#             ax.plot(cv_hist)
#         elif w == 'Numpy-01':
#             np_hist, bins = np.histogram(img_koala_rgb[:, :, i].ravel(), bins=256, range=(0, 256))
#             ax.plot(np_hist)
#         elif w == 'Numpy-02':
#             bc_hist = np.bincount(img_koala_rgb[:, :, i].ravel(), minlength=256)
#             ax.plot(bc_hist)
#         else:  # Matplotlib
#             ax.hist(img_koala_rgb[:, :, i].ravel(), bins=256, range=(0, 256))
#
#         ax.set_title(f"{w}通道({c})")
#
# plt.tight_layout()
# plt.show()

# fig = plt.figure("彩色直方图", figsize=(4, 8))
# # 使用枚举获取行索引
# for row_idx, w in enumerate(('CV', 'Numpy-01', 'Numpy-02', 'Matplotlib')):
#     # 全局索引 = 行索引 × 每行子图数 + 列索引 + 1
#     ax = fig.add_subplot(4, 2, row_idx * 2 + 1)
#     for i, c in enumerate(("b", "g", "r")):
#         if w == 'CV':
#             cv_hist = cv.calcHist([img_koala_rgb], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
#             ax.plot(cv_hist)
#         elif w == 'Numpy-01':
#             np_hist, bins = np.histogram(img_koala_rgb[:, :, i].ravel(), bins=256, range=(0, 256))
#             ax.plot(np_hist)
#         elif w == 'Numpy-02':
#             bc_hist = np.bincount(img_koala_rgb[:, :, i].ravel(), minlength=256)
#             ax.plot(bc_hist)
#         else:  # Matplotlib
#             ax.hist(img_koala_rgb[:, :, i].ravel(), bins=256, range=(0, 256))
#         ax.set_title(f"{w} BGR")
#
#     img_koala_hsv = cv.cvtColor(img_koala_rgb, cv.COLOR_BGR2HSV)
#
#     ax = fig.add_subplot(4, 2, row_idx * 2 + 2)
#     for i, c in enumerate(("h", "s", "v")):
#         if w == 'CV':
#             cv_hist = cv.calcHist([img_koala_hsv], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
#             ax.plot(cv_hist)
#         elif w == 'Numpy-01':
#             np_hist, bins = np.histogram(img_koala_hsv[:, :, i].ravel(), bins=256, range=(0, 256))
#             ax.plot(np_hist)
#         elif w == 'Numpy-02':
#             bc_hist = np.bincount(img_koala_hsv[:, :, i].ravel(), minlength=256)
#             ax.plot(bc_hist)
#         else:  # Matplotlib
#             ax.hist(img_koala_hsv[:, :, i].ravel(), bins=256, range=(0, 256))
#         ax.set_title(f"{w} HSV")
# plt.tight_layout()
# plt.show()

"""单独针对 H 通道（色调）进行统计计算"""

#
# # H 通道的直方图
# # channels=[0]：色调通道 [0, 179]
# # [1:, :]：去掉直方图的第一个 bin，以忽略大面积的无效色调（H=0）
# # 无效色调：饱和度为 0 的颜色，例如黑色、白色或灰色，此时 H 值无意义（通常被设为 0）
# img_xiaoren_hsv = cv.cvtColor(img_xiaoren_rgb, cv.COLOR_BGR2HSV)
# h_hist = cv.calcHist([img_xiaoren_hsv], channels=[0], mask=None, histSize=[256], ranges=[1, 256])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(h_hist)
#
# plt.show()

"""针对 HSV 图像计算直方图"""
# img_xiaoren_hsv = cv.cvtColor(img_xiaoren_rgb, cv.COLOR_BGR2HSV)
# # H -> 红色，S -> 黄色，V -> 绿色
# color = ('red', 'yellow', 'green')
# fig = plt.figure()
# for i, c in enumerate(color):
#     h_hist = cv.calcHist([img_xiaoren_hsv], channels=[i], mask=None, histSize=[256], ranges=[0, 256])[1:255, :]
#
#     print(img_xiaoren_rgb[i][233])
#     print(f"{i} min: {np.min(img_xiaoren_hsv[i])}, max: {np.max(img_xiaoren_hsv[i])}, {np.sum(img_xiaoren_hsv[i])}")
#
#     ax1 = fig.add_subplot(3, 2, i * 2 + 1)
#     ax1.plot(h_hist)
#     ax1.set_title(f"{c}直方图")
#
#
#     # 归一化直方图（消除数量级影响）
#     h_hist2 = h_hist / np.sum(h_hist)
#     ax2 = fig.add_subplot(3, 2, i * 2 + 2)
#     ax2.plot(h_hist2, color=c)
#     ax2.set_title(f"{c}归一化直方图")
#
# plt.show()

"""针对 HSV 图像计算直方图"""
#
# img_t_hsv = cv.cvtColor(img_t_rgb, cv.COLOR_BGR2HSV)
# h_hist = cv.calcHist([img_t_hsv], channels=[0], mask=None, histSize=[256], ranges=[0, 256])[1:, :]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.imshow(img_t_rgb)
# ax1.set_title("原图")
#
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(h_hist, color='red')
# ax2.set_title("H通道直方图")
#
# plt.show()


"""加入 mask 位置信息的直方图"""
# mask = np.zeros(img_koala.shape[0:2], dtype=np.uint8)
# mask[50:250, 50:450] = 255
#
# # 将图像与自身进行按位与运算，通过 mask 参数只保留掩码指定区域的像素。
# # src1 = img_koala_rgb (第一张图)
# # src2 = img_koala_rgb (第二张图，相同图像)
# # mask = 掩码图像（单通道，非0区域会被保留）
# # mask 之外的区域（mask=0）不进行任何运算，直接输出 0
# mask_img = cv.bitwise_and(img_koala, img_koala, mask=mask)
#
# hist1 = cv.calcHist([img_koala], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
# hist2 = cv.calcHist([mask_img], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 3)
# ax1.imshow(mask, cmap='gray')
# ax1.set_title("掩码")
#
# ax2 = fig.add_subplot(2, 2, 4)
# ax2.plot(hist2, color='green')
# ax2.plot(hist1, color='red')
# ax2.set_title("掩码区域直方图")
#
# ax3 = fig.add_subplot(2, 2, 1)
# ax3.imshow(img_koala, cmap='gray')
# ax3.set_title("原图")
#
# ax4 = fig.add_subplot(2, 2, 2)
# ax4.imshow(mask_img, cmap="gray")
# ax4.set_title("掩码图")
#
#
# plt.show()
# cv.imshow("mask_img", mask_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 转为 HSV 格式
# img = cv.cvtColor(img_t_rgb, cv.COLOR_BGR2HSV)
#
# # 创建一个 mask
# # 填充黑色，为屏蔽区域
# mask = np.zeros(img.shape[:2], np.uint8)
# # mask[95:300, 120:340] = 255  # 红灯
# mask[320:530, 120:340] = 255  # 黄灯
# # mask[550:770, 120:340] = 255  # 绿灯


# # 构建 mask 区域的图像
# masked_img = cv.bitwise_and(img, img, mask=mask)
#
# # 计算直方图
# hist1 = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])[1:, :]        # 不带掩膜
# hist1 = hist1 / np.sum(hist1)  # 归一化
# hist2 = cv.calcHist([masked_img], channels=[0], mask=mask, histSize=[256], ranges=[0,256])[1:, :]  # 带掩膜
# hist2 = hist2 / np.sum(hist2)
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(221)
# plt.imshow(cv.cvtColor(img, cv.COLOR_HSV2RGB))
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(cv.cvtColor(masked_img, cv.COLOR_HSV2RGB))  # 带 mask 的图像
# plt.title('masked_img')
#
# plt.subplot(223)
# plt.imshow(mask, 'gray')        # mask 图
# plt.title('mask')
#
# plt.subplot(224)
# plt.plot(hist1, color='red')    # 整个图像的 -> 绿色最多
# plt.plot(hist2, color='green')  # 仅计算 mask 区域的 -> 暴露的灯的颜色最多
# plt.title('Hist')
#
# plt.show()


"""加入 mask 位置信息的直方图"""

# # 加载图像
# # 转为 HSV 格式
# img = cv.cvtColor(img_t_rgb, cv.COLOR_BGR2HSV)
#
# # 黄色的范围
# lower = np.array([15, 50, 50])
# upper = np.array([35, 255, 255])
#
# # 绿色的范围
# # lower = np.array([40, 50, 50])
# # upper = np.array([60, 255, 255])
#
# # 在某个颜色范围内的掩膜
# mask = cv.inRange(img, lower, upper)
# print("图像形状：", img.shape)
# print("掩膜形状：", mask.shape)
#
# # 构建 mask 区域的图像
# masked_img = cv.bitwise_and(img, img, mask=mask)
#
# # 计算直方图
# hist1 = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])[1:, :]        # 不带掩膜
# hist1 = hist1 / np.sum(hist1)  # 归一化
# hist2 = cv.calcHist([masked_img], channels=[0], mask=mask, histSize=[256], ranges=[0,256])[1:, :]  # 带掩膜
# hist2 = hist2 / np.sum(hist2)
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(221)
# plt.imshow(cv.cvtColor(img, cv.COLOR_HSV2RGB))
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(cv.cvtColor(masked_img, cv.COLOR_HSV2RGB))  # 带 mask 的图像
# plt.title('masked_img')
#
# plt.subplot(223)
# plt.imshow(mask, 'gray')        # mask 图
# plt.title('mask')
#
# plt.subplot(224)
# plt.plot(hist1, color='red')    # 整个图像的 -> 绿色最多
# plt.plot(hist2, color='green')  # 仅计算 mask 区域的 -> 暴露的灯的颜色最多
# plt.title('Hist')
#
# plt.show()

"""
直方图均衡化
"""
# img_tsukuba_gray = cv.cvtColor(img_tsukuba_rgb, cv.COLOR_BGR2GRAY)
# # img_tsukuba_gray = cv.GaussianBlur(img_tsukuba_gray, (5, 5), 0)
#
# # 直方图均衡：增强图像的对比度
# # 通过重新分布图像的灰度级，使得像素值均匀分布在所有可能的范围内（0~255）
# # 从而让暗的区域更亮、亮的区域更暗，使图像细节更清晰
# equalizeHist = cv.equalizeHist(img_tsukuba_gray)
#
# # 自适应的直方图均衡（推荐）
# # 用于增强图像的局部对比度，同时避免传统直方图均衡化可能带来的过度增强问题
# # clipLimit=2.0：值越小，对比度增强越温和；值越大，对比度越强，但可能放大噪声
# # tileGridSize=(8, 8)：分块越小，局部对比度增强越细致，但计算量增大；分块越大，接近全局均衡化效果
#
# # 自适应的直方图均衡（推荐）
# # 用于增强图像的局部对比度，同时避免传统直方图均衡化可能带来的过度增强问题
# # clipLimit=2.0：值越小，对比度增强越温和；值越大，对比度越强，但可能放大噪声
# # tileGridSize=(8, 8)：分块越小，局部对比度增强越细致，但计算量增大；分块越大，接近全局均衡化效果
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 创建实例
# # 执行自适应的直方图均衡
# img3 = clahe.apply(equalizeHist)
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(221)
# plt.imshow(img_tsukuba_gray, 'gray')
# plt.title('Original Image')
#
# plt.subplot(222)
# plt.imshow(equalizeHist, 'gray')  # 直方图均衡
# plt.title('img2')
#
# plt.subplot(223)
# plt.imshow(img3, 'gray')  # 自适应的直方图均衡
# plt.title('img3')
#
# plt.subplot(224)
# plt.plot(cv.calcHist([img], [0], None, [256], [0, 256]), color='red', label='img')
# # plt.plot(cv.calcHist([img2], [0], None, [256], [0, 256]), color='green', label='img2')  # 非自适应的直方图均衡不太好
# plt.plot(cv.calcHist([img3], [0], None, [256], [0, 256]), color='blue', label='img3')
# plt.legend(loc='upper left')
# plt.title('Hist')
#
# plt.show()


"""
## 模板匹配
就是基于模板在原始图像中查找最匹配的位置
"""
from IPython.display import Markdown, display
methods = [
    'cv.TM_CCOEFF',  # 匹配系数
    'cv.TM_CCOEFF_NORMED',  # 归一化匹配系数
    'cv.TM_CCORR',  # 相关匹配法
    'cv.TM_CCORR_NORMED',  # 归一化相关匹配法
    'cv.TM_SQDIFF',  # 平方差匹配法
    'cv.TM_SQDIFF_NORMED'  # 归一化平方差匹配法
 ]

# 创建 Markdown 格式的表头
markdown_table1 = """
| 序号 | 匹配方式 | 左上角的点和右下角的点 | 相似度 | 类别 |
|------|------|------|------|------|
"""

fig = plt.figure(figsize=(20, 10))
w, h = img_template_gray.shape
for idx, method_cmd in enumerate(methods):
    img_xr_gray_copy = np.copy(img_xiaoren_gray)
    # 得到对应的方式（eval的意思是执行）
    method = eval(method_cmd)
    # 使用给定的方式进行模板匹配（返回值为各个局部区域和模板 template 之间的相似度）
    res = cv.matchTemplate(img_xr_gray_copy, img_template_gray, method)
    # 从数据中查找全局最小值、最大值，以及对应的位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    print("min_val: %s, max_val: %s, min_loc: %s, max_loc: %s" % (min_val, max_val, min_loc, max_loc))

    # 如果求解方式为 cv.TM_CCORR、TM_SQDIFF 或 TM_SQDIFF_NORMED，
    # 那么矩形左上角的点就是最小值的位置；
    # 否则是最大值的位置
    if method in [cv.TM_CCORR, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        # 基于左上角的坐标计算右下角的坐标
        # 分别加上模板图像的宽度和高度
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 打印：序号、匹配方式、左上角的点和右下角的点、相似度
    # 创建 Markdown 格式的表格内容
    markdown_table2 = f"""| {idx + 1} | {method_cmd} | {(top_left, bottom_right)} | {res[top_left[1], top_left[0]]} | {"最小值（黑色）" if top_left == min_loc else "最大值（白色）"} |
    """
    # 追加表格内容
    markdown_table1 += markdown_table2

    # 转换为 BGR（3 通道）
    bgr_img = cv.cvtColor(img_xiaoren_gray, cv.COLOR_GRAY2BGR)

    # 基于坐标画矩形
    cv.rectangle(bgr_img, top_left, bottom_right, color=[0, 0, 255], thickness=5)

    # 画图
    plt.subplot(len(methods) // 2, 4, 2 * idx + 1)  # 左图
    # 显示匹配结果的热力图（相似度数据可视化）
    # cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR_NORMED（右上角 3 个）：最大值（白色）最匹配，值越小越不匹配
    # cv.TM_CCORR, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED（左下角 3 个）：最小值（黑色）最匹配，值越大越不匹配
    plt.imshow(res, cmap='gray')
    plt.title(f'Matching Result {method_cmd}')
    plt.subplot(len(methods) // 2, 4, 2 * idx + 2)  # 右图
    plt.imshow(cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB))  # 原图（带矩形）
    plt.title(f'Detected Point {method_cmd}')

# 显示 Markdown 表格
display(Markdown(markdown_table1))

print()
plt.suptitle("Matching Result")  # 总标题
plt.show()