"""
学习放射变换过程
仿射变换的核心思想是求解M矩阵：
M矩阵的得到有两种方式：
1. 根据变换动作推到，比如要先旋转45度，缩放0.7，平移x方向：10，Y方向：14...
2. 根据变换前后三对点位进行反向计算得到，比如对地图进行矫正的时候，已知地图上三个点的真实坐标和地图坐标，就可以
计算出放射变换矩阵。
"""
import cv2 as cv
import numpy as np


def create_test_img():
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255 # 白色

    cv.rectangle(img, (100, 100), (250, 250), color=(255, 0, 0), thickness=-1)

    cv.circle(img, (100, 100), 8, color=(0, 0, 255), thickness=-1)
    cv.circle(img, (250, 200), 8, color=(0, 0, 255), thickness=-1)

    cv.putText(img, "Original", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 0, 0), 2)
    return img

def method1_known_params(img):
    """直接给出变换矩阵（最常用）"""
    angle = 30
    scale = 0.8
    tx, ty = 50, 30

    # 获取图像中心
    center = img.shape[0] // 2, img.shape[1] // 2
    # 方法1a：使用cv2.getRotationMatrix2D（只能处理旋转+缩放+平移）
    m = cv.getRotationMatrix2D(center, angle, scale)

    # 平移参数
    m[0, 2] += tx
    m[1, 2] += ty

    return cv.warpAffine(img, m, (img.shape[1], img.shape[0]))

def method2_from_points(img):
    """通过原图和目标图上指定三个点的对应关系来计算矩阵"""
    w, h = img.shape[1], img.shape[0]
    # 原始图像中的三个点（不共线）
    src_points = np.float32([
        [50, 50],  # 左上区域
        [w - 50, 50],  # 右上区域
        [50, h - 50]  # 左下区域
    ])

    # 变换后对应的三个点（实现错切效果）
    dst_points = np.float32([
        [80, 30],  # 左上向右下偏移
        [w - 30, 80],  # 右上向下偏移
        [30, h - 30]  # 左下向右偏移
    ])

    m = cv.getAffineTransform(src_points, dst_points)
    print(m)

    return cv.warpAffine(img, m, (w, h))

def method3_composite(img):
    """通过矩阵乘法组合多个变换"""
    # 定义基本变换矩阵（齐次形式 3x3）
    def translation(tx, ty):
        return np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)

    def rotation(angle, center=(0, 0)):
        """绕指定点旋转"""
        cx, cy = center

        # 先平移到原点，旋转，再平移回来
        t1 = translation(-cx, -cy)
        r = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        t2 = translation(cx, cy)

        return t1 @ r @ t2

    def scaling(sx, sy, center=(0, 0)):
        cx, cy = center
        t1 = translation(-cx, -cy)
        s = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        t2 = translation(cx, cy)

        return t1 @ s @ t2

    def shear(kx, ky):
        """错切"""
        return np.array([
            [1, kx, 0],
            [ky, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    # 组合变换：先缩放0.7倍，再旋转45度，再错切，再平移
    h, w = img.shape[0], img.shape[1]
    center = w // 2, h // 2
    m_total = translation(40, 20) @ \
    shear(0.3, 0) @ \
    rotation(np.radians(45), center) @ \
        scaling(0.7, 0.7, center)

    # 取前两行（warpAffine需要2x3矩阵）
    m_total = m_total[:2, :]
    print(m_total)
    return cv.warpAffine(img, m_total, (w, h))
if __name__ == '__main__':
    image = create_test_img()

    # image = method1_known_params(image)
    # image = method2_from_points(image)

    image = method3_composite(image)
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
