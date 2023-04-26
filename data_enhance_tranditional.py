import numpy as np
import random
import matplotlib.pyplot as plt
from load_fault_data import *


# 生成多参量时序数据
def generate_data(n_points=100, freq=1, noise=0.1):
    return load_fault_data()[0][23:, :]


# 噪声注入
def add_noise(data, noise_level=1):
    noisy_data = np.zeros_like(data)
    for i in range(9):
        noise = np.random.normal(0, noise_level, len(data))
        noisy_data[:, i] = data[:, i] + noise
    return noisy_data


def time_scale(data, scale):
    n_points = len(data)
    t_old = np.linspace(0, 1, n_points)
    t_new = np.linspace(0, 1, int(n_points*scale))
    scaled_data = np.interp(t_new, t_old, data)
    return scaled_data





# 测试噪声注入和时间扭曲
data = generate_data()
noisy_data = add_noise(data)
scaled_data = time_scale(data, 0.5)

fig, ax = plt.subplots()
# 遍历二维 ndarray 的每一列，画出对应的折线图
for i in range(9):
    ax.plot(data[:, i], label=labels[i])
# 设置 x 轴标签， y 轴标签， 标题和图例
ax.set_xlabel("date/d")
ax.set_ylabel("concentration/ppm")
ax.set_title('data')
ax.legend()
# 显示图表
plt.show()

fig, ax = plt.subplots()
# 遍历二维 ndarray 的每一列，画出对应的折线图
for i in range(9):
    ax.plot(noisy_data[:, i], label=labels[i])
# 设置 x 轴标签， y 轴标签， 标题和图例
ax.set_xlabel("date/d")
ax.set_ylabel("concentration/ppm")
ax.set_title('noisy_data')
ax.legend()
# 显示图表
plt.show()

fig, ax = plt.subplots()
# 遍历二维 ndarray 的每一列，画出对应的折线图
for i in range(9):
    ax.plot(scaled_data[:, i], label=labels[i])
# 设置 x 轴标签， y 轴标签， 标题和图例
ax.set_xlabel("date/d")
ax.set_ylabel("concentration/ppm")
ax.set_title('scaled_data')
ax.legend()
# 显示图表
plt.show()