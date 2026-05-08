""""
透视变换是仿射变换的推广，它能够处理近大远小的3D透视效果，而仿射变换做不到这一点。

透视变换有8个参数，为什么有8个参数？
3×3矩阵有9个元素，但可以整体缩放（齐次坐标特性），所以只有8个自由度
需要4个点（每个点提供2个方程）来确定变换
"""
import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

# img1 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png")
# h, w = img1.shape[:2]
# print(f"图像大小, 高度={h}, 宽度={w}")
#
# cv.line(img1, pt1=(0, h // 2), pt2=(w, h // 2), color=(255, 0, 0), thickness=2)
# cv.line(img1, pt1=(w // 2, 0), pt2=(w // 2, h), color=(255, 0, 0), thickness=2)
#
# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])  # 原图像：随机
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
#
# # 转换为整数（多边形的顶点坐标要求是整数）
# pts1_int = np.int32(pts1)
#
# # 画多边形
# cv.polylines(img1, [pts1_int], False, (0, 0, 255), 5)
#
# m = cv.getPerspectiveTransform(pts1, pts2)
# print("透视转换矩阵M：\n", m)
# dst = cv.warpPerspective(img1, m, (500, 500))
#
# cv.imshow("des", dst)
# cv.waitKey(0)

# plt.subplots(121)
# plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2GRAY))
# plt.title("原图")
#
# plt.subplots(122)
# plt.imshow(dst)
# plt.title("透视变换")
#
# plt.show()


# cv.imshow("img1", img1)
# cv.waitKey(0)
# cv.destroyAllWindows()


img = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/car3_plat.jpg")
h, w, _ = img.shape
print(f"图像大小, 高度={h}, 宽度={w}")

# 普通旋转逻辑：需要给定旋转中心 + 角度 + 缩放大小
# angle：正数表示逆时针旋转
M = cv.getRotationMatrix2D(center=(0, 0), angle=20, scale=1)
dst1 = cv.warpAffine(img, M, (w + 30, w // 2), borderValue=[0, 0, 0])
h, w, _ = dst1.shape

# 仿射变换：需要给定三个点在原始图像和新图像之间的坐标映射关系
pts1 = np.float32([[15, 0], [245, 0], [250, 115]])
pts2 = np.float32([[0, 0], [300, 0], [300, 115]])
# 构建对应的 M
M = cv.getAffineTransform(pts1, pts2)
print("仿射矩阵M：\n", M, sep="")
# 进行转换
dst2 = cv.warpAffine(dst1, M, (300, h))

# 透视变换：基于四个点在原始图像和新图像中的映射关系进行转换
pts1 = np.float32([[15, 0], [245, 0], [250, 115], [34, 100]])
pts2 = np.float32([[0, 0], [300, 0], [300, 115], [0, 115]])  # 业务给定
# 构建对应的 M
M = cv.getPerspectiveTransform(pts1, pts2)
print("\n透视矩阵M：\n", M, sep="")
dst3 = cv.warpPerspective(dst1, M, (300, h))

# 可视化画图
plt.subplot(141)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Input')
plt.subplot(142)
plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))  # 旋转图
plt.title('dst1')
plt.subplot(143)
plt.imshow(cv.cvtColor(dst2, cv.COLOR_BGR2RGB))  # 仿射图
plt.title('dst2')
plt.subplot(144)
plt.imshow(cv.cvtColor(dst3, cv.COLOR_BGR2RGB))  # 透视图
plt.title('dst3')
plt.show()