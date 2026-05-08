import PIL
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from pathlib import Path

img_path = Path(__file__).parent.parent.parent.parent / "assets/xiaoren.png"
img_xihu_path = Path(__file__).parent.parent.parent.parent / "assets/xihu.png"
img = Image.open(img_path)
img2 = img.copy()
# 查看图像信息：格式、尺寸(size=(宽,高))、模式(mode='RGB'/'L'等)
print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
# 打开图像
# img.show()
# 2. 图像变换
# 缩放（注意 thumbnail 会直接修改对象，不返回值）
resized_img = img.resize((100, 100))
# print(f"Format: {resized_img.format}, Size: {resized_img.size}, Mode: {resized_img.mode}")
# resized_img.show()
# img.show()
# 裁剪 (left, upper, right, lower)
crop_img = img.crop((0, 0, 100, 100))
# 旋转
rotate_img = img.rotate(45, expand=True) # expand=True 避免边缘被裁剪
rotate_img2 = img.rotate(45)
# rotate_img.show()
# rotate_img2.show()

# 3. 色彩与滤镜
# 转为灰度图 (mode='L'), HSV 模式
gray_img = img.convert(mode='L')
# gray_img.show()
# print(f"Format: {gray_img.format}, Size: {gray_img.size}, Mode: {gray_img.mode}")
# 应用模糊滤镜, 滤波器：RankFilter， 均值滤波器，中值滤波器，高斯滤波器
filter_img = img.filter(ImageFilter.RankFilter(7, 1))
# filter_img.show()
# 调整亮度 (系数>1变亮, <1变暗)
bright_img = ImageEnhance.Brightness(img).enhance(1.5)
dark_img = ImageEnhance.Brightness(img).enhance(0.5)
# bright_img.show()
# dark_img.show()

# 4. 绘图与文字
c_w, c_h = (img.width // 2, img.height // 2)
draw_img = ImageDraw.Draw(img)
draw_img.rectangle((c_w - 50, c_h - 50, c_w + 50, c_h + 50), fill=None, width=3, outline='red')
draw_img.line([(c_w, c_h), (c_w - 50, c_h - 50), (c_w + 50, c_h - 50)], fill='green', width=3)
draw_img.text((c_w, c_h), 'Hello World', fill='blue', size=25)
# img.show()

# 5. 粘贴与合并
# 将水印粘贴到底图 (第三个参数为遮罩，RGBA图片用自身Alpha通道)
main_img = Image.open(img_xihu_path)
main = main_img.copy()
logo_img = img.resize((100, 100), Image.LANCZOS).convert('RGBA')
main_img.paste(logo_img, (0, 0), logo_img)
# main_img.show() ## 如何粘贴透明图？？？

# 分离与合并通道
r, g, b, a = logo_img.split()
# r.show()
# g.show()
# b.show()
# a.show()
merged_img = Image.merge('RGB', (r, g, b))
# merged_img.show()

# 将图片转成透明图
logo_img2 = img.convert('RGB')
alpha = Image.new("L", logo_img2.size, 128)
logo_img2_copy = logo_img2.copy()
logo_img2_copy.putalpha(alpha)
main_img.paste(logo_img2_copy, (0, 0), logo_img2_copy)

# 将某种颜色变为透明（如纯白背景抠图）
logo_img_rgba = img2.convert('RGBA')
data = logo_img_rgba.get_flattened_data()
threshold_color = (240, 240, 240) # 近似白色
new_img_data = []
for item in data:
    if item[0] > threshold_color[0] and item[1] > threshold_color[1] and item[2] > threshold_color[2]:
        new_data = (threshold_color[0], threshold_color[1], threshold_color[2], 0)# A 通道设置为0，即透明
        new_img_data.append(new_data)
    else:
        new_img_data.append(item)

logo_img_rgba.putdata(new_img_data)
logo_img_rgba = logo_img_rgba.resize((100, 100))
logo_img_rgba.show()

# 3. 粘贴到主图（主图也需要是 RGBA 模式，否则透明无效）！！！
# main = main.convert("RGBA")
main.paste(logo_img_rgba, (0, 0), logo_img_rgba)
main.show()
# logo_img_rgba.show()
# logo_img2.show()
# logo_img2_copy.show()
# main_img.show()




# img.save("pillow_output.jpg", quality=85, optimize=True)
