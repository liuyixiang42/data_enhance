import numpy as np
import random
import matplotlib.pyplot as plt
from load_fault_data import *

add_noise_level = [2, 2, 1.7, 0.5, 0, 1.4, 4, 5, 2]
frequency_noise_levels = [15, 15, 13, 7, 0, 10, 36, 50, 45]

time_scale_typy = {'forward': 0, 'backward': 1}


# 生成多参量时序数据
def generate_data():
    return load_fault_data()[0]


def plt_data(data):
    fig, ax = plt.subplots()
    # 遍历二维 ndarray 的每一列，画出对应的折线图
    for i in range(9):
        ax.plot(data[:, i], label=labels[i])
    # 设置 x 轴标签， y 轴标签， 标题和图例
    ax.set_xlabel("date/d")
    ax.set_ylabel("concentration/ppm")
    ax.set_title('noisy_data')
    ax.legend()
    # 显示图表
    plt.show()


# 噪声注入
def add_noise(data):
    noisy_data = np.zeros_like(data)
    # 根据不同油气和油温的变化幅度采用不同的noise_level
    for i in range(9):
        noise = np.random.normal(0, add_noise_level[i], len(data))
        noisy_data[:, i] = data[:, i] + noise
    return noisy_data


def add_frequency_noise(data):
    # 将数据转换到频域
    freq_data = np.fft.fft(data, axis=0)
    # 增加噪声
    noise = np.zeros_like(freq_data)
    for i in range(data.shape[1] - 1):
        noise[:, i] = np.random.normal(scale=frequency_noise_levels[i], size=freq_data.shape[0])
    # 对最后一个维度油温不加噪声
    freq_data += noise
    # 将数据转换回时域
    return np.fft.ifft(freq_data, axis=0).real


def time_scale(data):
    type = np.random.randint(0, 2)
    if type == time_scale_typy['forward']:
        f_data = data[:30, :]
        b_data = data[30:, :]
        f_data = f_data[0::3, :]
        b_data = draw_single_data(b_data, 40)
        return np.concatenate((f_data, b_data), axis=0)
    elif type == time_scale_typy['backward']:
        f_data = data[:20, :]
        b_data = data[20:, :]
        f_data = draw_single_data(f_data, 40)
        b_data = b_data[0::3, :]
        return np.concatenate((f_data, b_data), axis=0)



data = generate_data()
plt_data(data)

noisy_data = add_noise(data)
plt_data(noisy_data)

f_data = add_frequency_noise(data)
plt_data(f_data)

time_scaled_data = time_scale(data)
plt_data(time_scaled_data)
