from pathlib import Path
import cv2 as cv
import numpy as np

img_j_path = Path('../assets/j.png')
img = cv.imread(str(img_j_path))
kernel = np.ones((5, 5), np.uint8)
img1 = cv.dilate(img, kernel, 1)
# 膨胀：图像中区域变大，空白区域变小
cv.imread('dilate', img1)
cv.imshow('img', img1)
cv.waitKey(-1)
# 腐蚀：图像中区域变小，空白区域变大
img2 = cv.erode(img, kernel, 1)
cv.imshow('img', img2)
cv.waitKey(-1)

# 相减
img3 = cv.subtract(img1, img2)
cv.imshow('img', img3)
cv.waitKey(-1)

# 开运算
img4 = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
cv.imshow('img', img4)
cv.waitKey(-1)

# 闭运算
img5 = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow('img', img5)
cv.waitKey(-1)

# 形态学梯度
img6 = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imshow('img', img6)
cv.waitKey(-1)

cv.destroyAllWindows()

