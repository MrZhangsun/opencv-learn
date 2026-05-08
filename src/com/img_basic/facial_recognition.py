import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

img_faces = cv.imread(Path(__file__).parent.parent.parent.parent.joinpath("assets/faces.png"))

# 加载定义好的人脸以及眼睛信息匹配信息
# 模型参数文件下载地址： https://github.com/opencv/opencv/tree/4.x/data/haarcascades
face_cascade = cv.CascadeClassifier(Path(__file__).parent.parent.parent.parent.joinpath("assets/model/haarcascade_frontalface_default.xml"))
eye_cascade = cv.CascadeClassifier(Path(__file__).parent.parent.parent.parent.joinpath("assets/model/haarcascade_eye.xml"))

# 转为灰度图
img_faces_gray = cv.cvtColor(img_faces, cv.COLOR_BGR2GRAY)

# 模板匹配中，模板大小固定，但图像中的人脸可能大小不一。解决方法是多次缩放图像，用固定窗口去滑动检测。
# 原始图像: 1920x1080
# ↓ 缩小 1/1.3
# 第1次缩放: 1477x831
# ↓ 缩小 1/1.3
# 第2次缩放: 1136x639
# ↓ 缩小 1/1.3
# 第3次缩放: 874x492
# ...
# 直到图像小于检测窗口(默认30x30)

"""
minNeighbors（最小邻域数）
作用：控制检测框的合并阈值，过滤误检
核心思想：图像中同一张脸可能被多个重叠窗口检测到，minNeighbors 决定需要多少个"邻居"窗口才认为是真脸
minNeighbors	效果	    优点	                缺点
2-3	            宽松	    检测率高，不漏检	    误检多（把非人脸的物体框出来）
5-6	            适中	    平衡准确率和召回率	    常用值
8-10        	严格	    误检极少	            可能漏检真人脸
"""
faces = face_cascade.detectMultiScale(img_faces_gray, scaleFactor=1.3, minNeighbors=5)
print("人脸数据的形状(人脸数量，坐标和尺寸)：", faces)

for (x, y, w, h) in faces:
    # 画人脸框
    cv.rectangle(img_faces, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # 获取人脸区域
    face_img = img_faces_gray[y:y + h, x:x + w]
    # 获取人脸区域中的眼睛
    eyes = eye_cascade.detectMultiScale(face_img)
    # 画眼睛框
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(img_faces, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)



cv.imshow("img_faces", img_faces)
cv.waitKey(0)
cv.imshow("img_faces_gray", img_faces_gray)
cv.waitKey(0)

cv.destroyAllWindows()

def face_detect(img_rgb):
    face_points = []
    eye_points = []

    """
    # 普通直方图均衡化（全局）
    img_equalized = cv.equalizeHist(gray_img)
    问题：整张图使用同一个变换，会导致：
        亮的地方过亮
        暗的地方过暗
        噪声被放大

    CLAHE 的优势：
        局部自适应：每个小区域单独均衡
        限制对比度：防止噪声过度放大
        适合光照不均的图像（如人脸侧光、医学图像）

    1. clipLimit（对比度限制）
    作用：限制直方图均衡化时的对比度放大程度
    clipLimit	效果	    优点          	缺点
    0-1     	弱增强	噪声不放大       	对比度改善不明显
    1.5-2.5 	适中增强	平衡对比度和噪声	推荐值
    3-5     	强增强	对比度提升明显	可能出现 artifacts
    >5      	过增强	细节夸张        	噪声严重放大，不自然

    2. tileGridSize（网格大小）
    作用：将图像分割成多个小区域，每个区域单独处理
    网格大小    	块大小	    局部性	速度	    适用场景
    (4, 4)  	大块	        弱局部	快   	光照缓慢变化
    (8, 8)  	中等      	适中  	中   	通用推荐
    (16, 16)	小块      	强局部	慢   	光照剧烈变化
    (32, 32)	很小      	很强  	很慢	    纹理细节增强
    """
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhance_img_gray = clahe.apply(img_gray)

    faces = face_cascade.detectMultiScale(enhance_img_gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # 添加人脸的坐标点
        face_points.append((x, y, (x + w), (y + h)))

        # 获得人脸区域
        face_area = img_gray[y:y + h, x:x + w]

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(face_area)

        for (ex, ey, ew, eh) in eyes:
            # 添加眼睛的坐标点
            eye_points.append((x + ex, y + ey, (x + ex + ew), (y + ey + eh)))

    return face_points, eye_points

def draw_rect(img, face_points, eye_points):

    # 画人脸框
    for (tx, ty, bx, by) in face_points:
        cv.rectangle(img,  (tx, ty), (bx, by), (255, 0, 0), 2)

    for (tx, ty, bx, by) in eye_points:
        cv.rectangle(img, (tx, ty), (bx, by), (0, 255, 0), 2)

    return img

# 检测
def detect_show(img_faces):
    face_points, eye_points = face_detect(img_faces)
    img_faces = draw_rect(img_faces, face_points, eye_points)
    cv.imshow("img_faces", img_faces)
    cv.waitKey(0)
    cv.destroyAllWindows()


"""从摄像机获取视频 + 人脸区域提取"""
cap = cv.VideoCapture(0)
success, frame = cap.read()
while success and cv.waitKey(1) != ord('q'):
    success, frame = cap.read()
    # 检测
    if success:
        face_points, eye_points = face_detect(frame)
        img = draw_rect(frame, face_points, eye_points)
        cv.imshow("img", img)

        if len(faces) > 0:
            # NOTE: 检测出人脸后，直接将人脸发送给服务器进行业务逻辑 + 模型预测等相关处理
            pass

cap.release()
cv.destroyAllWindows()