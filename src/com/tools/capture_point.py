import cv2


class CapturePointTool(object):
    """
    鼠标左键点击获取坐标
    """
    def __init__(self, path):
        self.points = []
        self.img_path = path

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x}, {y})")
            self.points.append((x, y))

    def capture_point(self):
        img = cv2.imread(self.img_path)
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", self.mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("捕获到的点：", self.points)
        return self.points










