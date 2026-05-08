""""
形态学转换

通过结构核的形状，对元素逐个进行运算，根据不同的转换规则，得到当前元素变换后的值。
结构核通过二维矩阵进行定义，1：表示结构核形状，0:表示空白，不用于覆盖元素范围。

形态学转换的过程：
1. 定义结构核
2. 确定转换算法：腐蚀、膨胀、开运算、闭运算、形态梯度、顶帽、黑帽
3. 对齐像素，逐个计算，得到当前元素转换后的值

转换算法：
腐蚀：只有当 kernel 覆盖区域“全是1”，中心点（当前点）才保留1
    效果：
        去掉小白点噪声
        让边界收缩
    类比：像被“吃掉”了一圈
膨胀：只要 kernel 覆盖区域有白色，中心点就变白
    效果：
        填补空洞
        扩大目标
    类比：像“长胖”了一圈
开运算：先腐蚀，再膨胀
    用途：
        去除小噪声（小白点）
        保留整体形状
闭运算：先膨胀，再腐蚀
    用途：
        填补小黑洞
        连接断裂区域
形态梯度：边缘提取，本质：梯度 = 膨胀 - 腐蚀
    效果：得到物体轮廓（类似边缘检测）
顶帽：公式 = 原图 - 开运算，提取亮的小目标
黑帽：公式 = 闭运算 - 原图，提取暗的小区域

| 操作  | 本质       |
| --- | -------- |
| 腐蚀  | 去掉边缘、缩小  |
| 膨胀  | 扩展边缘、变大  |
| 开运算 | 去噪（去小白点） |
| 闭运算 | 填洞（补黑洞）  |
| 梯度  | 找轮廓      |


API:
dst3 = cv.morphologyEx(img, op={type}, kernel=kernel, iterations=1)  # 形态梯度
type:
cv.MORPH_ERODE: 腐蚀
cv.MORPH_DILATE: 膨胀
cv.MORPH_OPEN: 开运算
cv.MORPH_CLOSE: 闭运算
cv.MORPH_GRADIENT: 形态梯度
"""
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
import numpy as np

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

# 定义结构核
kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, ksize=(5, 5))
kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(5, 5))
kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
print("矩形kernel:\n", kernel_rect, sep="")
print("十字kernel:\n", kernel_cross, sep="")
print("椭圆kernel:\n", kernel_ellipse, sep="")

img_j = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/j.png"))

"""
腐蚀：用结构核与图像像素对齐，逐个元素开始计算结构核覆盖区域内的像素的最小值min作为当前像素的像素值
"""
# # 进行腐蚀操作
# # a. 定义一个核（全部设置为 1 表示对核中 5*5 区域的所有像素均进行考虑，设置为 0 表示不考虑）
# # 核的定义和卷积不一样，卷积里面是参数的意思，腐蚀里面是范围的意思（1 表示考虑，0 表示不考虑）
# dst_rect = cv.erode(img_j, kernel_rect)
# dst_cross = cv.erode(img_j, kernel_cross)
# dst_ellipse = cv.erode(img_j, kernel_ellipse)
# # c. 可视化
# cv.imshow('img', img_j)
# cv.imshow('dst_rect', dst_rect)  # 腐蚀操作
# cv.imshow('dst_cross', dst_cross)  # 腐蚀操作
# cv.imshow('dst_ellipse', dst_ellipse)  # 腐蚀操作
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
膨胀
和腐蚀的操作相反，其功能是增加图像的白色区域的值
1. 滑动内核：将内核的中心（锚点，默认是中心点）对准输入图像上的每一个像素。
2. 计算最大值：查看内核所覆盖的输入图像区域，取出这个区域内所有像素值的最大值。
3. 替换锚点像素：用这个计算出的最大值来替换输出图像中锚点位置的像素值。

通常情况下，在去除噪声后，可以通过膨胀再恢复图像的目标区域信息。
"""

# dst_rect = cv.dilate(img_j, kernel_rect)
# dst_cross = cv.dilate(img_j, kernel_cross)
# dst_ellipse = cv.dilate(img_j, kernel_ellipse)
#
# cv.imshow('img', img_j)
# cv.imshow('dst_rect', dst_rect)
# cv.imshow('dst_cross', dst_cross)
# cv.imshow('dst_ellipse', dst_ellipse)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
Open 开运算
Open 其实指的就是先做一次腐蚀，然后再做一次膨胀操作，一般用于去除白色噪声。
"""

# # 加载噪声数据
# h, w, _ = img_j.shape
# for i in range(100):
#     y = np.random.randint(w)
#     x = np.random.randint(h)
#
#     img_j[x, y] = 255 # 灰度图，加白点
#
# dst_rect = cv.morphologyEx(img_j, cv.MORPH_OPEN, kernel_rect, iterations=1)
# dst_cross = cv.morphologyEx(img_j, cv.MORPH_OPEN, kernel_cross, iterations=1)
# dst_ellipse = cv.morphologyEx(img_j, cv.MORPH_OPEN, kernel_ellipse, iterations=1)
#
# # 模拟验证
#
# # b. 先腐蚀
# dst1 = cv.erode(img_j, kernel_rect, iterations=1)
#
# # c. 再膨胀
# dst2 = cv.dilate(dst1, kernel_rect, iterations=1)
#
# print(dst2[139, 33])
# print(dst_rect[139, 33])

# cv.imshow('img', img_j)
# cv.imshow('dst_rect', dst_rect)
# cv.imshow('dst_rect_manul', dst2)
# cv.imshow('dst_cross', dst_cross)
# cv.imshow('dst_ellipse', dst_ellipse)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Closing 闭运算
Closing 其实指的就是先做一次膨胀，再做一次腐蚀；<br>
对前景图像中的如果包含黑色点，有一共去除的效果。
"""
# 加载噪声数据
# h, w, _ = img_j.shape
# for i in range(100):
#     y = np.random.randint(w)
#     x = np.random.randint(h)
#
#     img_j[x, y] = 255 # 灰度图，加白点
#
# # 加黑色点
# for i in range(1000):
#     y = np.random.randint(w)
#     x = np.random.randint(h)
#     img_j[x, y] = 0  # 灰度图，加黑点
#
# dst_rect = cv.morphologyEx(img_j, cv.MORPH_CLOSE, kernel_rect, iterations=1)
# dst_cross = cv.morphologyEx(img_j, cv.MORPH_CLOSE, kernel_cross, iterations=1)
# dst_ellipse = cv.morphologyEx(img_j, cv.MORPH_CLOSE, kernel_ellipse, iterations=1)
#
# # 模拟验证
#
# # b. 先膨胀
# # dst1 = cv.dilate(img_j, kernel_rect, iterations=1)
# dst1 = cv.morphologyEx(img_j, cv.MORPH_DILATE, kernel_rect, iterations=1)
#
# # c. 再腐蚀
# # dst2 = cv.erode(dst1, kernel_rect, iterations=1)
# dst2 = cv.morphologyEx(dst1, cv.MORPH_ERODE, kernel_rect, iterations=1)
#
# print(dst2[139, 33])
# print(dst_rect[139, 33])
#
# cv.imshow('img', img_j)
# cv.imshow('dst_rect', dst_rect)
# cv.imshow('dst_rect_manul', dst2)
# cv.imshow('dst_cross', dst_cross)
# cv.imshow('dst_ellipse', dst_ellipse)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Morphological Gradient 形态梯度
就是在膨胀和腐蚀之间的操作，也就是在膨胀的图像和腐蚀的图像之间取差集，<br>
一般的结果就是边缘位置显示，其他位置不显示（类似提取边缘，但不是）；

一般做这个之前，先做一个噪声数据去除的操作。
"""
#
# dst_erode_rect = cv.morphologyEx(img_j, cv.MORPH_ERODE, kernel_rect, iterations=1)
# dst_dilate_rect = cv.morphologyEx(img_j, cv.MORPH_DILATE, kernel_rect, iterations=1)
# dst_gradient_rect = cv.morphologyEx(img_j, cv.MORPH_GRADIENT, kernel_rect, iterations=1)
#
# cv.imshow('img', img_j)
# cv.imshow('dst_erode_rect', dst_erode_rect)
# cv.imshow('dst_dilate_rect', dst_dilate_rect)
# cv.imshow('dst_gradient_rect', dst_gradient_rect)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Top Hat
在原始图像和Open图像之间获取差集，提取一些非交叉点的信息。一般不用。
"""
# # 调整一下结构核，以便效果明显
# kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, ksize=(9, 9))
#
# # Open（先腐蚀，再膨胀）
# dst_open_rect = cv.morphologyEx(img_j, cv.MORPH_OPEN, kernel_rect, iterations=1)
#
# # Top Hat(原始图 - open图)
# dst_tophat_rect = cv.morphologyEx(img_j, cv.MORPH_TOPHAT, kernel_rect, iterations=1)
#
# cv.imshow('img', img_j)
# cv.imshow('dst_open_rect', dst_open_rect)
# cv.imshow('dst_tophat_rect', dst_tophat_rect)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Black Hat
在原始图像和 Close 操作图像之间取差集，提取一些交叉点附近的特征信息。一般不用。
"""
# # 调整一下结构核，以便效果明显
# kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, ksize=(9, 9))
#
# # Close(先膨胀，再腐蚀)
# dst_close_rect = cv.morphologyEx(img_j, cv.MORPH_CLOSE, kernel_rect, iterations=1)
#
# # Black Hat(close图 - 原始图)
# dst_blackhat_rect = cv.morphologyEx(img_j, cv.MORPH_BLACKHAT, kernel_rect, iterations=1)
#
# cv.imshow('img', img_j)
# cv.imshow('dst_close_rect', dst_close_rect)
# cv.imshow('dst_blackhat_rect', dst_blackhat_rect)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
实践操作：通过形态学变换进行车牌处理
"""
img_car = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/car.jpg"))
# 转成灰度图
img_car_gray = cv.cvtColor(img_car, cv.COLOR_BGR2GRAY)
# 高斯模糊（0：根据 ksize 自动计算标准差）
img_car_blur = cv.GaussianBlur(img_car_gray, (5, 5), 0)

# b. 形态梯度(dilate - erode)，显示边缘位置
plate1 = cv.morphologyEx(img_car_blur, cv.MORPH_GRADIENT, kernel_rect, iterations=1)

# 二值化处理
ret, plate2 = cv.threshold(plate1, 127, 255, cv.THRESH_BINARY)

kernel = np.asarray([
    [1, 1, 1, 1, 1, 1, 1, 1]
])

# plate3 = cv.morphologyEx(plate2, cv.MORPH_DILATE, kernel, iterations=1)
plate3 = cv.morphologyEx(plate2, cv.MORPH_CLOSE, kernel, iterations=1)


cv.imshow('img_car_blur', img_car_blur)
cv.imshow('img_car_gray', img_car_gray)
cv.imshow('img_car', img_car)
cv.imshow('plate1', plate1)
cv.imshow('plate2', plate2)
cv.imshow('plate3', plate3)
cv.waitKey(0)
cv.destroyAllWindows()



