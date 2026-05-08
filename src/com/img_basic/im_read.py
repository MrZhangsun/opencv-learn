import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

"""
图像的本质都是一个多维数组，可以用numpy进行读取和操作，数组格式：[H,W,C]
H：图像高度
W：图像宽度
C：颜色通道
"""
img_path = Path(__file__).parent.parent.parent.parent / "assets/car.jpg"
img = cv.imread(img_path)
print(type(img)) # <class 'numpy.ndarray'>图像数据就是一个numpy数组
print(img.shape) # (377, 550, 3) -> 高度，宽度，通道数
print(img.size)
print(img.dtype)
print(img[1,1,:])
print(img[:,:,0]) # 获取所有图片的B通道数据
print(img[:,:,1]) # 获取所有图片的G通道数据
print(img[:,:,2]) # 获取所有图片的R通道数据

# 图像保存
# cv.imwrite('save_car.png', img, [cv.IMWRITE_PNG_COMPRESSION, 9])

# plt.imshow(img, cmap='gray') # matploylib显示图片
# plt.show()
cv.imshow('car', img)
cv.waitKey(-1)
cv.destroyAllWindows()