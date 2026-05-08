# 图像的算数运算
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from fontTools.unicodedata import block
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题
# img_path = Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png"
# img = cv.imread(img_path)
# print(img.shape) # (600, 510, 3)
# cv.imshow("xiaoren", img)
# cv.waitKey(-1)
#
# """
# 显示三原色
# 要显示三原色，必须是三通道，单通道就成灰度图了
# """
# img2_b = img[:,:,0] # B G R
# img2_g = img[:,:,1] # B G R
# img2_r = img[:,:,2] # B G R
#
# """
# cv.merge() 的核心要求
# 要求：所有通道必须满足
# shape 完全一致（H, W）
# dtype 一致（如 uint8）
# """
# padding = np.zeros_like(img2_b) # 创建一个和img一样的全零矩阵
# img2_b = cv.merge([img2_b, padding, padding])
# img2_g = cv.merge([padding, img2_g, padding])
# img2_r = cv.merge([padding, padding, img2_r])
#
# cv.imshow("img2_b", img2_b)
# cv.waitKey(0)
# cv.imshow("img2_g", img2_g)
# cv.waitKey(0)
# cv.imshow("img2_r", img2_r)
# cv.waitKey(0)
#
# """
# 图像裁剪
# 只需要裁剪H，W，不能裁剪通道C
# """
# img100_100 = img[0:100, 0:100]
# cv.imshow("img100_100", img100_100)
# cv.waitKey(0)
#
# """
# 访问图像的像素
# """
# px = img[250, 300]
# print(f"位置(250, 300)对应的像素为：{px}")
# px_blue = img[250, 300, 0]
# print(f"位置(250, 300)对应的像素的蓝色取值为：{px_blue}")
# print("位置(250, 300)对应的像素蓝色取值为：", img.item(250, 300, 0))
# img[250, 300, 0] = 100
# print("位置(250, 300)对应的像素新的蓝色取值为：", img.item(250, 300, 0))
#
# """
# 修改某个通道的颜色
# """
# img_copy = np.copy(img)
# img_copy[:, :, 2] = 127
# print(img_copy)
# print(f"位置(250, 300)对应的像素为：{img[250, 300]}")
# cv.imshow("img_red_127", img_copy)
# cv.waitKey(0)
# cv.imshow("raw_img", img)
# cv.waitKey(0)
#
# """
# 粘贴图像
# """
# box1 = img[0:95, 20:240]
# box2 = img[0:95, 280:500]
# box3 = box2 * 0.7 +  box1 * 0.3
# img_copy2 = np.copy(img)
# img_copy2[0:95, 280:500] = box3 # 替换/粘贴
# cv.imshow("img_paste", img_copy2)
# cv.waitKey(0)
#
# """
# 图像通道的分割和合并
# """
# b, g, r = cv.split(img)
# img_rgb = cv.merge([r, g, b])
# cv.imshow("img_rgb", img_rgb)
# cv.waitKey(0)

"""
添加边框
"""
# img_logo_path = Path(__file__).parent.parent.parent.parent / "assets/opencv-logo.png"
# img_logo = cv.imread(img_logo_path)
# print("logo原图大小：", img_logo.shape)
# # cv.imshow("logo原图", img_logo)
# # cv.waitKey(0)
#
# # 复制并添加边框
# border_constant = cv.copyMakeBorder(img_logo, 10, 10, 10, 10,
#                                          borderType=cv.BORDER_CONSTANT, value=[128, 128, 128])
# print("添加边框后图大小：", border_constant.shape)
# # cv.imshow("border_constant", border_constant)
# # cv.waitKey(0)
# # 边界反射
# border_reflect = cv.copyMakeBorder(img_logo, 10, 10, 10, 10,
#                                    borderType=cv.BORDER_REFLECT)
# print("边界反射图大小：", border_reflect.shape)
#
# # 边界延伸循环
# border_wrap = cv.copyMakeBorder(img_logo, 10, 10, 10, 10,
#                   borderType=cv.BORDER_WRAP)
#
# # 边界反射 101（边界像素不保留）
# border_reflect_101 = cv.copyMakeBorder(img_logo, 10, 10, 10, 10,
#                   borderType=cv.BORDER_REFLECT_101)
# img_6c = np.arange(54).reshape((3, 3, 6))
# print(img_6c)
#
# img_6c_replicate = cv.copyMakeBorder(img_6c, 3, 3, 3, 3,
#                   borderType=cv.BORDER_REPLICATE)
#
# print("\n带框图的形状（0轴和1轴分别增加了3x2=6个像素）：\n", img_6c_replicate.shape, sep="")
# print("\n原图换轴后（每个矩阵是一个通道）：\n", np.transpose(img_6c, (2, 0, 1)).shape, sep="")
# print("\n带框图换轴后（每个矩阵是一个通道）：\n", np.transpose(img_6c_replicate, (2, 0, 1)).shape, sep="")
#
# # 展示对比图
# fig = plt.figure(figsize=(20, 10))
# gs = GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1],
#          hspace=0.35, wspace=0.3)
# fig.suptitle("原图和边框对比")
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[0, 3])
# ax5 = fig.add_subplot(gs[1, 0])
# ax6 = fig.add_subplot(gs[1, 1])
#
# ax1.set_title("原图")
# ax2.set_title("border_constant")
# ax3.set_title("border_reflect")
# ax4.set_title("border_wrap")
# ax5.set_title("border_reflect_101")
#
# # 将数据画到坐标系里
# ax1.imshow(img_logo)
# ax2.imshow(border_constant)
# ax3.imshow(border_reflect)
# ax4.imshow(border_wrap)
# ax5.imshow(border_reflect_101)
# print("图片显示")
# plt.show()


"""
图像合并
"""
# img1 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png")
# img2 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/opencv-logo.png")

# 设置为相同大小
# dsize=(300, 300)：目标尺寸 (width, height)
# img1 = cv.resize(img1, dsize=(300, 300))
# img2 = cv.resize(img2, dsize=(300, 300))

# 添加背景
# 使用.addWeighted() 函数对两张图像进行加权融合，自动会进行归一化处理
# 计算公式：dst = alpha * src1 + beta * src2 + gamma
# gamma=-80 # 黑色
# alpha = 0.3
# beta = 1.0
#
# img3 = cv.addWeighted(img1, alpha, img2, beta, gamma)
# cv.imshow("img3", img3)
# cv.waitKey(0)
#
# # 加权融合
# img3 = img1 * alpha + img2 * beta + gamma
# # 归一化 大于255的取255，小于0的取0
# img3 = np.clip(img3, 0, 255)
# print(img3)
# cv.imshow("img3", img3)
# cv.waitKey(0)

# img2 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/opencv-logo.png")
# img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# cv.imshow("img2_gray", img2_gray)
# cv.waitKey(0)

"""
图像二值化
图像像素值大于thresh的数值，数值设置为：maxval，小于thresh的设置为0
二值化类型：
THRESH_BINARY：黑白二值图
THRESH_BINARY_INV：黑白反过来
THRESH_TRUNC：截断亮度（> thresh → thresh <= thresh → 原值）
THRESH_TOZERO：去掉暗区域（> thresh → 原值<= thresh → 0）
THRESH_TOZERO_INV：去掉亮区域（> thresh → 0 <= thresh → 原值）
"""
# thresh, img2_thresh_binary = cv.threshold(img2_gray, thresh=10, maxval=255, type=cv.THRESH_BINARY)
# cv.imshow("img2_binary", img2_thresh_binary)
# cv.waitKey(0)
#
# _, img2_thresh_binary_inv = cv.threshold(img2_gray, thresh=10, maxval=255, type=cv.THRESH_BINARY_INV)
# cv.imshow("img2_thresh_binary_inv", img2_thresh_binary_inv)
# cv.waitKey(0)
#
# _, img2_thresh_trunc = cv.threshold(img2_gray, thresh=10, maxval=255, type=cv.THRESH_TRUNC)
# cv.imshow("img2_thresh_trunc", img2_thresh_trunc)
# cv.waitKey(0)
#
# _, img2_thresh_tozero = cv.threshold(img2_gray, thresh=10, maxval=255, type=cv.THRESH_TOZERO)
# cv.imshow("img2_thresh_tozero", img2_thresh_tozero)
# cv.waitKey(0)
#
# _, img2_thresh_tozero_inv = cv.threshold(img2_gray, thresh=10, maxval=255, type=cv.THRESH_TOZERO_INV)
# cv.imshow("img2_thresh_tozero_inv", img2_thresh_tozero_inv)
# cv.waitKey(0)


"""
图像的位运算（将 logo 放到图像的中上方）

"""
img1 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png")
img2 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/opencv-logo.png")
# rows1, cols1, _ = img1.shape
# rows2, cols2, _ = img2.shape
#
# start_row = 50
# end_row = start_row + rows2
#
# start_col = 0
# end_col = cols2
#
# roi = img1[start_row:end_row, start_col:end_col]
# cv.rectangle(img1, pt1=(start_col, start_row), pt2=(end_col, end_row),
#              color=(0, 0, 255), thickness=2)
# # 将图像转换为灰度图像
# img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# # 将灰度图像转换为黑白图像，做一个二值化操作
# thresh, img2_thresh_binary = cv.threshold(img2_gray, thresh=10, maxval=255, type=cv.THRESH_BINARY)
# # 对图像做一个求反的操作，即 255-mask(白底、黑图像)
# mask_inv = cv.bitwise_not(img2_thresh_binary)
# cv.imshow("img2_thresh_binary", img2_thresh_binary)
# cv.waitKey(0)
#
# cv.imshow("mask_inv", mask_inv)
# cv.waitKey(0)
#
# print(roi.shape)
# print(mask_inv.shape)
# print(img1.shape)
# print(img2.shape)
# 获取得到背景图
# 对两个相同尺寸的图像进行按位与运算，并应用反向掩膜控制操作区域
# 在求解 bitwise_and 操作的时候，如果给定 mask，只对 mask 中对应为白色的位置进行 and 操作，其它位置直接设置为 0

# img3_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# img2_fg = cv.bitwise_and(img2, img2, mask=img2_thresh_binary)
# cv.imshow("img3_bg", img3_bg)
# cv.waitKey(0)
# cv.imshow("img2_fg", img2_fg)
# cv.waitKey(0)
#
# img4 = cv.add(img3_bg, img2_fg)
# cv.imshow("img4", img4)
# cv.waitKey(0)
# img1[start_row:end_row, start_col:end_col] = img4
#
# cv.imshow("img1", img1)
# cv.waitKey(0)


"""
标准的图像融合
将一张图片和另外一张图片进行融合
"""
# rows1, cols1, _ = img1.shape
# rows2, cols2, _ = img2.shape
#
# start_row = 50
# end_row = start_row + rows2
#
# start_col = 0
# end_col = cols2
# roi = img1[start_row:end_row, start_col:end_col]
#
# img2_resized = cv.resize(img2, (roi.shape[1], roi.shape[0]))
#
# gray = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)
# _, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
#
# mask_inv = cv.bitwise_not(mask)
#
# bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# fg = cv.bitwise_and(img2_resized, img2_resized, mask=mask)
#
# dst = cv.add(bg, fg)
#
# img1[start_row:end_row, start_col:end_col] = dst
#
# cv.imshow("img1", img1)
# cv.waitKey(0)

'''
图像旋转
'''
# dst = cv.rotate(img1, cv.ROTATE_90_CLOCKWISE)
# cv.imshow("img1", dst)
# cv.waitKey(0)
#
# dst = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
# cv.imshow("img1", dst)
# cv.waitKey(0)
#
# dst = cv.rotate(img1, cv.ROTATE_180)
# cv.imshow("img1", dst)
# cv.waitKey(0)

"""
图像反转
flipCode=0 垂直翻转
flipCode=1 水平翻转
"""

img1 = cv.flip(img1, flipCode=1)
cv.imshow("img1", img1)
cv.waitKey(0)

# 垂直
v = cv.flip(img1, flipCode=0)
cv.imshow("v", v)
cv.waitKey(0)

# 水平
h = cv.flip(img1, flipCode=1)
cv.imshow("h", h)
cv.waitKey(0)


cv.destroyAllWindows()



