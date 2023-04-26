import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column = ['氢气', '一氧化碳', '甲烷', '乙烯', '乙炔', '乙烷', '总烃', '总可燃气体', '油温']
labels = ['Hydrogen', 'Carbon Monoxide', 'Methane', 'Ethylene', 'Acetylene', 'Ethane', 'Total Hydrocarbons',
          'Total Combustible Gases', 'Oil Temperature']
fault_category = ['insulation aging', 'winding fault', 'insulation breakdown', 'overload fault', 'short-circuit fault',
                  'cooling fault']


def draw_data(fault_data):
    # 将原始数据长度调整为目标长度的倍数
    new_fault_data = []
    length = 50
    for j in range(6):
        data = fault_data[j]
        data = np.repeat(data, length // data.shape[0], axis=0)

        # 线性插值
        for i in range(data.shape[1]):
            for t in range(length):
                t0 = t // 10 * 10
                t1 = t0 + 10
                y0 = data[t0 // 10, i]
                y1 = data[t1 // 10, i]
                data[t, i] = y0 + (y1 - y0) / 10 * (t - t0)
        new_fault_data.append(data)
    new_fault_data = np.array(new_fault_data)
    return new_fault_data


def load_fault_data():
    fault_data = pd.read_excel('故障样本.xlsx', usecols=column)

    fault_data.dropna(inplace=True)

    fault_data = fault_data.values

    fault_data = fault_data[:173, :]

    rows_to_delete = [144, 115, 86, 57, 28]
    fault_data = np.delete(fault_data, rows_to_delete, axis=0)
    fault_data = fault_data.reshape(6, 28, 9)
    fault_data = fault_data[:, 23:, :]
    fault_data = fault_data.astype(np.float64)

    return draw_data(fault_data)
    # return fault_data


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
