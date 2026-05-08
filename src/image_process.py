import cv2 as cv
import numpy as np
from src.com.tools.capture_point import CapturePointTool
from pathlib import Path

image_path = '../assets/xiaoren.png'

# img = cv.imread(image_path)
# print(img.shape)
# w = img.shape[1]
# h = img.shape[0]
# M = cv.getRotationMatrix2D((w / 2, h / 2), 90, 0.8)
#
# # 仿射变换：对图像做“拉伸 + 旋转 + 平移 + 倾斜”，但保持“直线还是直线”
# img = cv.warpAffine(img, M, (w + 30, w // 2), borderValue=[0, 0, 0])
# # img = cv.flip(img, 1) # 1: 水平翻转，0: 垂直反转
# cv.imshow('img', img)
# cv.waitKey(-1)
# cv.destroyAllWindows()

cwd = Path(__file__).resolve().parent.parent
car_plat_path = f'{cwd}/assets/car3_plat.jpg'

# car_plat = cv.imread(car_plat_path)
# cv.imshow('car_plat', car_plat)
# from_point = []
# to_point = []

#
# pts1 = np.float32([[50,50], [200,50], [50,200]])
# pts2 = np.float32([[10,100], [200,50], [100,250]])
#
# M = cv.getAffineTransform(pts1, pts2)
# cv.warpAffine(car_plat, M)
# cv.waitKey(-1)
# cv.destroyAllWindows()


# capture_point = CapturePointTool(car_plat_path)
# points = capture_point.capture_point()
# # 车牌四个角（需要你手动或算法检测）
# src = np.float32([(22, 13), (222, 87), (199, 195), (1, 104)])
# # 目标位置（正面矩形）
# dst = np.float32([(20, 12), (217, 12), (220, 90), (9, 81)])
#
# M = cv.getPerspectiveTransform(src, dst)
#
# result = cv.warpPerspective(car_plat, M, (300, 100))
#
# cv.imshow("result", result)
# cv.waitKey(0)
