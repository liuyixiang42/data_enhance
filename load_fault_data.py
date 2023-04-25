import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

column = ['氢气', '一氧化碳', '甲烷', '乙烯', '乙炔', '乙烷', '总烃', '总可燃气体', '油温']
labels = ['Hydrogen', 'Carbon Monoxide', 'Methane', 'Ethylene', 'Acetylene', 'Ethane', 'Total Hydrocarbons',
          'Total Combustible Gases', 'Oil Temperature']
fault_category = ['insulation aging', 'winding fault', 'insulation breakdown', 'overload fault', 'short-circuit fault',
                  'cooling fault']


def load_fault_data():
    data = pd.read_excel('故障样本.xlsx', usecols=column)

    data.dropna(inplace=True)

    data = data.values

    data = data[:173, :]

    rows_to_delete = [144, 115, 86, 57, 28]
    data = np.delete(data, rows_to_delete, axis=0)
    data = data.reshape(6, 28, 9)
    return data


if __name__ == 'main':
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
