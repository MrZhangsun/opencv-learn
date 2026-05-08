"""
图像变换：缩放、旋转、裁剪


"""
import cv2 as cv
import numpy as np
from pathlib import Path

img1 = cv.imread(Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png")
"""
大小重置: cv.resize
"""
# old_h, old_w, _ = img1.shape
# print(f"旧图像的大小, 高度={old_h}, 宽度={old_w}")
#
# new_h = int(old_h * 0.8)
# new_w = 250
# dst = cv.resize(img1, (new_w, new_h))
# print(f"新图像的大小, 高度={new_h}, 宽度={new_w}")

"""
仿射变换：cv.

"""
M = np.float32([
    [1, 0, 20],
    [0, 1, -10]
])
cv.warpAffine(img1, M, (img1.shape[1], img1.shape[0]), borderValue=[0, 0, 0])
