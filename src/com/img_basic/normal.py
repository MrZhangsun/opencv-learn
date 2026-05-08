import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-10, 10, 1000)
plt.rcParams['font.sans-serif'] = ['PingFang HK']  # mac常用中文字体
plt.rcParams['axes.unicode_minus'] = False         # 解决负号显示问题

# 不同均值的正态分布
mus = [-3, 0, 2]
colors = ['red', 'blue', 'green']
labels = [f'μ = {mu}, σ = 1' for mu in mus]

plt.figure(figsize=(10, 6))
for mu, color, label in zip(mus, colors, labels):
    y = norm.pdf(x, mu, 1)
    plt.plot(x, y, color=color, label=label, linewidth=2)
    # 标记均值位置
    plt.axvline(x=mu, color=color, linestyle='--', alpha=0.5)

plt.xlabel('x', fontsize=12)
plt.ylabel('概率密度', fontsize=12)
plt.title('均值的几何意义：曲线的对称轴', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()