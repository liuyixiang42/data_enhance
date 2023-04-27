import numpy as np
import random
import matplotlib.pyplot as plt
from load_fault_data import *

add_noise_level = [1, 1, 0.8, 0.2, 0, 0.5, 1.5, 1.6, 1.1]
frequency_noise_levels = [7, 7, 5, 2, 0, 3, 14, 15, 11]

noise_factor = [8, 7, 6, 1, 1, 4, 10, 25, 60]

time_scale_typy = {'forward': 0, 'backward': 1}


# 生成多参量时序数据
def generate_data():
    return load_fault_data()[[0, 4]]


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
        factor = data[:, i] / noise_factor[i]
        for j in range(len(factor)):
            if factor[j] > 2.5:
                factor[j] = 2.5
        noise = factor * noise
        noisy_data[:, i] = data[:, i] + noise

    # 将负值替换成0
    noisy_data[noisy_data < 0] = 0
    return noisy_data


def add_frequency_noise(data):
    # 将数据转换到频域
    freq_data = np.fft.fft(data, axis=0)
    # 增加噪声
    noise = np.zeros_like(freq_data)
    for i in range(data.shape[1]):
        noise[:, i] = np.random.normal(scale=frequency_noise_levels[i], size=freq_data.shape[0])
        factor = data[:, i] / noise_factor[i]
        for j in range(len(factor)):
            if factor[j] > 2.9:
                factor[j] = 2.9
        noise[:, i] = noise[:, i] * factor

    freq_data += noise
    # 将数据转换回时域
    noisy_data = np.fft.ifft(freq_data, axis=0).real

    # 将负值替换成0
    noisy_data[noisy_data < 0] = 0
    return noisy_data

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



for i in range(2):
    plt_data(data[i])

    noisy_data = add_noise(data[i])
    plt_data(noisy_data)

    f_data = add_frequency_noise(data[i])
    plt_data(f_data)

    time_scaled_data = time_scale(data[i])
    plt_data(time_scaled_data)

    tmp_data = time_scale(data[i])
    tmp_data = add_noise(tmp_data)
    tmp_data = add_frequency_noise(tmp_data)
    plt_data(tmp_data)


