import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的边缘：左边暗(0)，右边亮(255)
img = np.zeros((1, 10), dtype=np.float64)
img[0, 5:] = 255

# 手动计算 Laplacian (简化的一维情况)
# 二阶差分：f(x+1) - 2*f(x) + f(x-1)
laplacian = np.zeros_like(img)
for i in range(1, 9):
    laplacian[0, i] = img[0, i+1] - 2*img[0, i] + img[0, i-1]

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
axes[0].plot(img[0], 'b-', linewidth=2)
axes[0].set_title('Original Signal (Edge: dark→light)')
axes[0].set_ylabel('Intensity')

axes[1].plot(laplacian[0], 'r-', linewidth=2)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_title('Laplacian Response (note: positive and negative)')
axes[1].set_ylabel('Laplacian Value')

# 解释正负的含义
axes[2].plot(np.abs(laplacian[0]), 'g-', linewidth=2)
axes[2].set_title('Absolute Value (lost directional info)')
axes[2].set_ylabel('|Laplacian|')
axes[2].set_xlabel('Pixel Position')

plt.tight_layout()
plt.show()

print(f"Laplacian 值: {laplacian[0]}")
print(f"在边缘左侧: {laplacian[0, 4]:.1f} (负值)")
print(f"在边缘右侧: {laplacian[0, 5]:.1f} (正值)")