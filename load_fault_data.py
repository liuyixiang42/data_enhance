import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

column = ['氢气', '一氧化碳', '甲烷', '乙烯', '乙炔', '乙烷', '总烃', '总可燃气体', '油温']
labels = ['Hydrogen', 'Carbon Monoxide', 'Methane', 'Ethylene', 'Acetylene', 'Ethane', 'Total Hydrocarbons',
          'Total Combustible Gases', 'Oil Temperature']
fault_category = ['insulation aging', 'winding fault', 'insulation breakdown', 'overload fault', 'gas leakage',
                  'cooling fault']


def smooth_and_extend(seq, length):
    # 定义原始序列
    x = seq
    # 定义插值的x坐标轴
    x_new = np.linspace(0, len(x) - 1, num=length, endpoint=True)
    # 进行样条插值
    cs = CubicSpline(np.arange(len(x)), x)
    smooth = cs(x_new)
    # 确保起始值和终值不变
    smooth[0] = x[0]
    smooth[-1] = x[-1]
    return smooth


def draw_data(fault_data, length):
    new_fault_data = []
    for i in range(6):
        new_data = []
        data = fault_data[i]
        for j in range(9):
            seq = data[:, j]
            new_seq = smooth_and_extend(seq, length)
            new_data.append(new_seq)
        new_fault_data.append(new_data)
    return np.transpose(new_fault_data, (0, 2, 1))


def draw_single_data(fault_data, length):
    new_data = []
    data = fault_data
    for j in range(9):
        seq = data[:, j]
        new_seq = smooth_and_extend(seq, length)
        new_data.append(new_seq)
    return np.array(new_data).T


def load_fault_data():
    fault_data = pd.read_excel('故障样本.xlsx', usecols=column)

    fault_data.dropna(inplace=True)

    fault_data = fault_data.values

    fault_data = fault_data[:191, :]

    rows_to_delete = [159, 127, 95, 63, 31]
    fault_data = np.delete(fault_data, rows_to_delete, axis=0)
    fault_data = fault_data.reshape(6, 31, 9)
    fault_data = fault_data[:, 20:, :]
    fault_data = fault_data.astype(np.float64)

    return draw_data(fault_data, 50)


if __name__ == '__main__':
    data = load_fault_data()
    for j in range(6):
        # 创建图表对象
        fig, ax = plt.subplots()
        # 遍历二维 ndarray 的每一列，画出对应的折线图
        for i in range(9):
            ax.plot(data[j][:, i], label=labels[i])

        # 设置 x 轴标签， y 轴标签， 标题和图例
        ax.set_xlabel("date/d")
        ax.set_ylabel("concentration/ppm")
        ax.set_title(fault_category[j])
        ax.legend()

        # 显示图表
        plt.show()


