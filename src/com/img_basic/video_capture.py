import cv2 as cv

capture = cv.VideoCapture(0)
w = capture.get(cv.CAP_PROP_FRAME_WIDTH)
h = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
size = (int(w), int(h))
print("视频宽高：" + str(size))
print("视频帧率：" + str(capture.get(cv.CAP_PROP_FPS)))
success, frame = capture.read()

fps = 25
# video_writer = cv.VideoWriter("output.avi", cv.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)
video_writer = cv.VideoWriter(
    "output.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    size
)
frame_num = 6

num_frames_remaining = 6* fps - 1
while success and  num_frames_remaining > 0:
    frame_num += 1
    success, frame = capture.read()
    print(f"the {frame_num} frame, type: %s, shape: %s, size: %d" % (type(frame), frame.shape, frame.size))

    # 保存视频
    video_writer.write(frame)
    num_frames_remaining -= 1
capture.release()
cv.destroyAllWindows()

