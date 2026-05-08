"""
图像轮廓
轮廓 ≠ 边缘
轮廓和边缘不是一回事！ 这是图像处理中非常容易混淆的概念。让我详细解释它们的区别。

特性	轮廓 (Contour)	边缘 (Edge)
定义	物体边界的完整闭合曲线	像素强度突变的点
数学本质	区域边界的点集（拓扑结构）	梯度极值点（微分特征）
是否闭合	✅ 必须闭合	❌ 不一定闭合（可以是线段）
组织方式	有层次结构（父子嵌套）	独立点集，无组织
依赖条件	需要二值图像（前景/背景明确）	可直接在灰度图上计算
输出格式	有序点集 [(x1,y1), (x2,y2), ...]	二值图像（边缘点为白色）
抗噪能力	强（基于区域连通性）	弱（基于局部梯度）


轮廓信息可以简单的理解为图像曲线的连接点信息，在目标检测以及识别中有一定的作用。

轮廓信息的查找最好是基于灰度图像或者边缘特征图像，因为基于这样的图像比较容易找连接点信息；<br/>
NOTE: 在 OpenCV 中，查找轮廓是在黑色背景中查找白色图像的轮廓信息。
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

img_xiaoren = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/xiaoren.png"))
img_car3_plat = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/car3_plat.jpg"))
img_car3 = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/car3.jpg"))

# cv.imshow("xiaoren", img_xiaoren)
# cv.imshow("xiaoren1", img_xiaoren1)
# cv.imshow("xiaoren2", img_xiaoren2)
# cv.waitKey(0)
# cv.destroyAllWindows()
"""
轮廓和边缘的对比
"""
# # 创建一个简单的图像：一个带缺口的圆
# img = np.zeros((300, 300), dtype=np.uint8)
# cv.circle(img, (150, 150), 100, 255, -1)  # 实心圆
#
# # 在圆上开个缺口（制造不闭合的边缘）
# img[140:160, 240:260] = 0
#
# # 1. 边缘检测（Canny）
# edges = cv.Canny(img, 50, 150)
#
# # 2. 轮廓检测（需要二值图像）
# contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#
# # 可视化对比
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Original (with gap)')
# axes[0].set_xlabel('白色区域有缺口，不是闭合区域')
#
# axes[1].imshow(edges, cmap='gray')
# axes[1].set_title('Canny Edges')
# axes[1].set_xlabel('检测到所有边界点，包括缺口处的端点')
#
# # 绘制轮廓
# img_contour = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# cv.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
# axes[2].imshow(cv.cvtColor(img_contour, cv.COLOR_BGR2RGB))
# axes[2].set_title('Contours')
# axes[2].set_xlabel('找不到闭合轮廓！因为区域不闭合')
#
# plt.tight_layout()
# plt.show()
#
# print(f"边缘检测到的像素数: {np.sum(edges > 0)}")
# print(f"轮廓数量: {len(contours)}")

"""
查找轮廓
步骤：
1. 图像转灰度图
2. 图像转黑底图（非必须）
3. 二值化
4. 查找轮廓
"""
# img_xiaoren1 = cv.cvtColor(img_xiaoren, cv.COLOR_BGR2GRAY)
# # 做一个图像反转 (0 -> 255, 255 -> 0)
# # 在 OpenCV 中，查找轮廓是在黑色背景中查找白色图像的轮廓信息。
# img_xiaoren2 = cv.bitwise_not(img_xiaoren1)
# # 做一个二值化
# ret, img_xiaoren3 = cv.threshold(img_xiaoren2, 127, 255, cv.THRESH_BINARY)
#
# # 发现轮廓信息
# # cv.RETR_TREE：保留完整层次结构（适合复杂嵌套）
# # CHAIN_APPROX_NONE：保留所有边界点（精度最高，内存最大）
# # contours：每个轮廓的坐标点信息
# # hierarchy: 轮廓之间的关系
# contours, hierarchy = cv.findContours(img_xiaoren3, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# print(f"轮廓数量: {len(contours)}")
# print(f"轮廓关系: {hierarchy}")
# print(type(contours))  # <class 'tuple'>
# print(contours[0])
# print(hierarchy[0].shape)

# # hierarchy 的形状: (1, 轮廓数量, 4)
# # 每个轮廓对应一个包含4个整数的数组
# print(hierarchy.shape)  # (1, N, 4)
#
# # 对于第 i 个轮廓，其层级关系为：
# hierarchy[0][i] = [Next, Previous, First_Child, Parent]

# # 将轮廓绘制出来
# img_copy = np.copy(img_xiaoren)
# # cv.drawContours(img_copy, contours, -1, (0, 0, 255), 2)
#
# # cv.imshow("contours", img_copy)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# # 计算点最多的轮廓的索引
# contour_points_count = [len(c) for c in contours]
# max_contour_idx = max_point_contour = np.argmax(contour_points_count)
# cv.drawContours(img_copy, contours, max_contour_idx, (0, 255, 0), 2)
#
# # cv.imshow("contours", img_copy)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# # 创建黑色图像
# img1 = np.zeros_like(img_xiaoren)
# cv.drawContours(img1, contours, -1, (255, 255, 255), 2)
# # cv.imshow("contours", img1)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# # 可视化
# plt.figure(figsize=(20, 10))
#
# plt.subplot(231)
# plt.imshow(cv.cvtColor(img_xiaoren, cv.COLOR_BGR2RGB))
# plt.title('Original Image')
#
# plt.subplot(232)
# plt.imshow(img_xiaoren3, cmap = 'gray')  # 二值化
# plt.title('thresh')
#
# plt.subplot(233)
# plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))  # 在图像中绘制轮廓
# plt.title('img3')
#
# plt.subplot(234)
# plt.imshow(img_xiaoren1, cmap = 'gray')  # 灰度图
# plt.title('img1')
#
# plt.subplot(235)
# plt.imshow(img_xiaoren2, cmap = 'gray')  # 反转后的灰度图
# plt.title('img2')
#
# plt.subplot(236)
# plt.imshow(img1, cmap = 'gray')  # 将轮廓当成边缘
# plt.title('img4')
#
# plt.show()

"""
查找车牌号轮廓
"""
# # 转换为灰度图
# img1 = cv.cvtColor(img_car3_plat, cv.COLOR_BGR2GRAY)
# # 做图像反转 (0 -> 255, 255 -> 0)
# img2 = cv.bitwise_not(img1)
# # 二值化
# ret, img3 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
# # 发现轮廓信息
# # cv.RETR_LIST：检测所有轮廓但不建立层级关系
# # CHAIN_APPROX_SIMPLE：对于一条直线上的点而言，仅仅保留端点信息
# # contours：每个轮廓的坐标点信息
# # hierarchy: 轮廓之间的关系
# contours, hierarchy = cv.findContours(img3, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# # 绘制轮廓
# img_car3_plat_copy = np.copy(img_car3_plat)
# # # contourIdx=-1：绘制所有轮廓
# # cv.drawContours(img_car3_plat_copy, contours, -1, (0, 0, 255), 2)
#
# # # 绘制点最多的轮廓
# length_contours = [len(contour) for contour in contours]
# max_contour_idx = np.argmax(length_contours)
# cv.drawContours(img_car3_plat_copy, contours, max_contour_idx, (0, 255, 0), 2)
#
# # 将轮廓当成边缘（白线）
# img4 = np.zeros_like(img_car3_plat)
# # contourIdx=-1：绘制所有轮廓
# cv.drawContours(img4, contours, -1, (255, 255, 255), 2)
#
# # 显示
# fig = plt.figure("车牌轮廓", figsize=(20, 10))
# ax11 = fig.add_subplot(2, 3, 1, title="原图")
# ax11.imshow(cv.cvtColor(img_car3_plat, cv.COLOR_BGR2RGB))
#
# ax12 = fig.add_subplot(2, 3, 2, title="灰度")
# ax12.imshow(img1, cmap='gray')
#
# ax13 = fig.add_subplot(2, 3, 3, title="反转")
# ax13.imshow(img2, cmap='gray')
#
# ax14 = fig.add_subplot(2, 3, 4, title="二值")
# ax14.imshow(img3, cmap='gray')
#
# ax15 = fig.add_subplot(2, 3, 5, title="轮廓")
# ax15.imshow(cv.cvtColor(img_car3_plat_copy, cv.COLOR_BGR2RGB))
#
# ax16 = fig.add_subplot(2, 3, 6, title="轮廓")
# ax16.imshow(img4, cmap='gray')
#
# plt.show()
# # cv.imshow("img_car3_plat", img_car3_plat)
# # cv.imshow("img1", img1)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
"""
构建黑底白框的图像（1 个通道）
"""
#
# # 构建黑底白框的图像（1 个通道）
# img = np.zeros((300, 300), np.uint8)
# img[10:290, 10:290] = 255
# img[50:200, 50:200] = 0
# img[55:100, 55:100] = 255
# img[120:190, 120:150] = 255
# img[130:160, 130:145] = 0
# img[210:250, 210:250] = 0
# img[250:270, 150:180] = 0
# img[205:220, 205:220] = 0
#
# # 拷贝图像（二值化）
# print(img.shape)
# img_copy = np.copy(img)
#
# # 发现轮廓信息
# # cv.RETR_TREE：检测所有轮廓并建立完整的层级树
# # CHAIN_APPROX_SIMPLE：对于一条直线上的点而言，仅仅保留端点信息
# # contours：每个轮廓的坐标点信息
# # hierarchy: 轮廓直接的关系
# contours, hierarchy = cv.findContours(img_copy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
# # 转换为 BGR 格式（3 个通道）
# img_bgr = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)
# # 在图像中绘制轮廓
# # contourIdx=idx：绘制索引为 idx 的轮廓
#
# for idx in range(len(contours)):
#     if idx // 3 == 0:
#         cv.drawContours(img_bgr, contours, idx, (0, 0, 255), 2)
#     elif idx // 3 == 1:
#         cv.drawContours(img_bgr, contours, idx, (255, 0, 0), 2)
#     else:
#         cv.drawContours(img_bgr, contours, idx, (0, 255, 255), 2)
# # 在图像中添加文本
# # org=(10, 10)：宽度、高度
# cv.putText(img_bgr, text='0', org=(10, 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
# cv.putText(img_bgr, text='1', org=(150, 250), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_AA)
# cv.putText(img_bgr, text='2', org=(205, 205), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_AA)
# cv.putText(img_bgr, text='3', org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_AA)
# cv.putText(img_bgr, text='4', org=(120, 120), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_AA)
# cv.putText(img_bgr, text='5', org=(130, 130), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_AA)
# cv.putText(img_bgr, text='6', org=(65, 55), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv.LINE_AA)
# # 在图像中添加文本
# # org=(10, 10)：宽度、高度
# fig = plt.figure("模拟图轮廓查找")
# ax11 = fig.add_subplot(1, 2, 1, title="原图")
# ax11.imshow(img, cmap='gray')
#
# ax12 = fig.add_subplot(1, 2, 2, title="轮廓")
# ax12.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
#
# plt.show()

"""
轮廓属性
当获取得到轮廓坐标后，就可以基于轮廓来计算面积、周长等属性。
"""
#
# img_xiaoren_gray = cv.cvtColor(img_xiaoren, cv.COLOR_BGR2GRAY)
#
# img_xiaoren_gray_ivn = cv.bitwise_not(img_xiaoren_gray)
#
# ret, img_xiaoren_bin = cv.threshold(img_xiaoren_gray_ivn, 127, 255,
#                                cv.THRESH_BINARY)
#
# # 发现轮廓信息
# # RETR_LIST：检索所有轮廓，但是不保留层次信息
# # HAIN_APPROX_SIMPLE：对于一条直线上的点而言，仅仅保留端点信息
# # contours：每个轮廓的坐标点信息
# # hierarchy: 轮廓之间的关系
# contours, hierarchy = cv.findContours(img_xiaoren_bin, cv.RETR_LIST,
#                                       cv.CHAIN_APPROX_SIMPLE)
# # 坐标点最多的轮廓的索引
# max_idx = np.argmax([len(contour) for contour in contours])
# print(f"最大轮廓的索引为：{max_idx}")
# # 选择一个坐标点最多的轮廓进行处理
# max_contour = contours[max_idx]
#
# cv.drawContours(img_xiaoren, contours, max_idx, (0, 255, 0), 2)
# print("轮廓的形状(坐标点数量, 1, 坐标)：", max_contour.shape, sep="")
# print("\n轮廓数组（宽度，高度）：\n", max_contour, sep="")
#
# # cv.imshow("contours", img_xiaoren)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
# # 计算面积
# cv.contourArea(max_contour)
# print("轮廓的面积为：", cv.contourArea(max_contour))
# # 计算周长
# cv.arcLength(max_contour, True)
# print("轮廓的周长为：", cv.arcLength(max_contour, True))
# # 获取最大的矩形边缘框，返回值为矩形框的左上角的坐标，以及宽度和高度
# x, y, h, w = cv.boundingRect(max_contour)
# print("矩形边缘框的坐标为：", x, y, h, w)
#
# cv.rectangle(img_xiaoren, (x, y), (x + h, y + w), (220, 0, 0), 2)
# #
# # cv.imshow("contours", img_xiaoren)
# # cv.waitKey(0)
# # cv.destroyAllWindows()

"""绘制最小矩形（所有边缘在矩形内）"""
#
# # 得到矩形的点（左下角、左上角、右上角、右下角<顺序不一定>）、绘图
#
# # minAreaRect：求得一个包含点集 cnt 的最小面积的矩形，这个矩形可以有一点旋转的，输出为矩形的四个坐标点
# # rect为三元组：
# # 第 1 个元素为旋转中心点的坐标
# # 第 2 个元素为矩形的宽度和高度
# # 第 3 个元素为旋转大小，正数表示顺时针旋转，负数表示逆时针旋转
# min_rect = cv.minAreaRect(max_contour)
# print("最小矩形的坐标为：", min_rect)
#
# # 从 旋转矩形 中提取其 4 个角点坐标
# box_points = cv.boxPoints(min_rect)
# print("旋转矩形的 4 个角点坐标为：", box_points)
# # 转成整数
# box_points = np.uint64(box_points)
#
# # cv.drawContours(img_xiaoren, [box_points], 0, (0, 0, 255), 2)
# cv.polylines(img_xiaoren, [box_points], True, (0, 0, 255), 2)
# #
# cv.imshow("contours", img_xiaoren)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""绘制最小的圆（所有边缘在圆内）"""
# # minEnclosingCircle：求得一个包含点集 cnt 的最小面积的圆形，输出为圆形中心点的坐标以及半径
# (x, y), r = cv.minEnclosingCircle(max_contour)
# print("最小圆的坐标为：", (x, y), "半径为：", r)
#
# # 绘制圆
# cv.circle(img_xiaoren, (int(x), int(y)), int(r), (0, 0, 255), 2)
# # 可视化
# cv.imshow('img', img_xiaoren)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""绘制最小的椭圆"""
# # 所有边缘不一定均在圆内，一般不用
# ellipse = cv.fitEllipse(max_contour)
# print("最小椭圆：", ellipse)
#
# # 绘制圆
# cv.ellipse(img_xiaoren, ellipse, color=(0, 0, 255), thickness=2)
#
# # 可视化
# cv.imshow('img', img_xiaoren)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
旋转后提取最小矩形
"""
# # 旋转
# # rect为三元组：
# # 第 1 个元素为旋转中心点的坐标
# # 第 2 个元素为矩形的宽度和高度
# # 第 3 个元素为旋转大小，正数表示顺时针旋转，负数表示逆时针旋转
# # ((x, y), (w, h), angle)
# # ((302.3465576171875, 419.5862731933594), (372.9076232910156, 317.0608215332031), 70.12313079833984)
# # 顺时针旋转 70 度，意味着矩形逆时针旋转 20 度
# rect = [(302, 420), (373, 317), -20]
#
# # 放射变换的M矩阵
# M = cv.getRotationMatrix2D(center=rect[0], angle=rect[2], scale=1)
# # 从灰度图的形状中提取高度、宽度
# h, w, _ = img_xiaoren.shape
#
# img_xiaoren = cv.cvtColor(img_xiaoren, cv.COLOR_BGR2RGB)
# # 仿射变换
# # borderValue=[255, 255, 255]：边界填充白色
# img_affine = cv.warpAffine(img_xiaoren, M, (w, h), borderValue=(255, 255, 255))
#
# img_gray = cv.cvtColor(img_affine, cv.COLOR_BGR2GRAY)
# img_gray_ivn = cv.bitwise_not(img_gray)
# ret, img_bin = cv.threshold(img_gray_ivn, 127, 255, cv.THRESH_BINARY)
# contours, hierarchy =cv.findContours(img_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# # 选择一个坐标点最多的轮廓进行处理
# max_idx = np.argmax([len(contour) for contour in contours])
# max_contour = contours[max_idx]
#
# # 绘制轮廓
# # contourIdx=idx：绘制索引为 idx 的轮廓
# cv.drawContours(img_xiaoren, contours, max_idx, (0, 255, 0), 2)
#
# # 绘制最小矩形（所有边缘在矩形内）
# # minAreaRect：求得一个包含点集 cnt 的最小面积的矩形，这个矩形可以有一点的旋转的
# # rect为三元组：
# # 第 1 个元素为旋转中心点的坐标
# # 第 2 个元素为矩形的宽度和高度
# # 第 3 个元素为旋转大小，正数表示顺时针旋转，负数表示逆时针旋转
# min_rect = cv.minAreaRect(max_contour)
#
# box_points = cv.boxPoints(min_rect)
# box_points = np.uint64(box_points)
#
# cv.drawContours(img_xiaoren, [box_points], 0, (0, 0, 255), 2)
# print("最小矩形((x, y), (w, h), angle)：\n", min_rect, sep="")
# print("\n从旋转矩形中提取其4个角点坐标：\n", box_points, sep="")
#
#
#
#
# fig = plt.figure("旋转后提取最小矩形", figsize=(8, 8))
# ax11 = fig.add_subplot(1, 3, 1, title="原图")
# ax11.imshow(img_xiaoren, cmap='gray')
# ax12 = fig.add_subplot(1, 3, 2, title="仿射变换")
# ax12.imshow(img_affine, cmap='gray')
# ax13 = fig.add_subplot(1, 3, 3, title="二值化")
# ax13.imshow(img_bin)
#
# plt.show()


# 读取图像，car3.jpg
# 转成灰度图
img = cv.cvtColor(img_car3, cv.COLOR_BGR2GRAY)
# 高斯平滑去噪，核 (5, 5)，标准差 2
img = cv.GaussianBlur(img, (5, 5), 2)

# 边缘检测（返回二值化图像）
# threshold1=50, threshold2=250：低阈值、高阈值
img1 = cv.Canny(img, threshold1=50, threshold2=250)

# 膨胀操作
# 形状：(2, 5)，垂直方向膨胀少点，水平方向膨胀多点
kernel = np.asarray([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
])
img2 = cv.dilate(img1, kernel, iterations=3)

# 轮廓查找
# RETR_LIST：检索所有轮廓，但是不保留层次信息
# HAIN_APPROX_SIMPLE：对于一条直线上的点而言，仅仅保留端点信息
# contours：每个轮廓的坐标点信息
# hierarchy: 轮廓直接的关系
contours, hierarchy = cv.findContours(img2, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# 拷贝图像
img3 = np.copy(img2)

# 转换为 BGR 格式
bgr_img = cv.cvtColor(img3, cv.COLOR_GRAY2BGR)

# 绘制轮廓
# contourIdx=-1：绘制所有轮廓
cv.drawContours(bgr_img, contours, contourIdx=-1, color=[0, 0, 255], thickness=2)

# 可视化
cv.imshow('img', img)
cv.imshow('img1', img1)        # 边缘检测
cv.imshow('img2', img2)        # 膨胀操作
cv.imshow('bgr_img', bgr_img)  # 绘制轮廓
cv.waitKey(0)
cv.destroyAllWindows()


"""处理图像中的轮廓，筛选出符合条件的轮廓，并对其进行旋转校正和可视化"""

img5 = np.copy(img3)  # 拷贝图像
rows, cols = img5.shape  # 提取高度、宽度

cont_idx = 0  # 初始化轮廓索引

# 针对所有找到的轮廓进行遍历
for cnt in contours:
    cont_idx += 1  # 放在这里，是因为排除不符合要求的矩形后也能累加
    rect = cv.minAreaRect(cnt)  # 最小外接矩形 ((x, y), (w, h), angle)
    w, h = rect[1]  # 最小外接矩形的宽度和高度

    if w < 50 or h < 50:  # 排除过小的矩形
        continue
    r = w / h
    if r > 4 or r < 1 / 4:  # 排除长宽比异常的矩形
        continue
    area = cv.contourArea(cnt)  # 轮廓区域的面积
    if area < 10000:  # 排除面积过小的轮廓
        continue
    area2 = w * h  # 最小外接矩形的面积

    # 旋转
    # 生成一个 2D 旋转矩阵
    M = cv.getRotationMatrix2D(center=rect[0], angle=rect[-1], scale=1)
    # 仿射变换
    # borderValue=[255, 255, 255]：边界填充白色
    img6 = cv.warpAffine(img5, M, (cols, rows), borderValue=[255, 255, 255])

    print("\n=========================")
    print(f"轮廓区域的面积：{int(area)}")
    print(f"最小外接矩形的面积：{int(area2)}")
    print(f"面积比（轮廓区域/最小外接矩形）：{area / area2:.2f}")  # 选择面积比最大的，例如大于 50%，可能就是我们要找的区域
    print("最小外接矩形((x, y), (w, h), angle)：\n", rect, sep="")

    # 从 旋转矩形 中提取其 4 个角点坐标
    box = cv.boxPoints(rect)
    # 转成整数
    box = np.int64(box)
    print("\n旋转矩形的4个角点坐标：\n", box, sep="")

    # 转换为 BGR（3 通道）
    bgr_img = cv.cvtColor(img5, cv.COLOR_GRAY2BGR)

    # 绘制轮廓
    # contourIdx=cont_idx - 1：绘制第 cont_idx - 1 个轮廓
    cv.drawContours(bgr_img, contours, contourIdx=cont_idx - 1, color=[0, 0, 255], thickness=2)

    # 可视化
    cv.imshow('bgr_img', bgr_img)  # 绘制轮廓后的图像
    cv.imshow('img6', img6)  # 仿射变换后的图像
    cv.waitKey(0)
    cv.destroyAllWindows()