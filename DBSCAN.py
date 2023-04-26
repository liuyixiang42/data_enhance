from sklearn.preprocessing import MinMaxScaler
from load_fault_data import *
from sklearn.mixture import GaussianMixture

def normalize3(a):
    for i in range(a.shape[0]):
        min_a, max_a = np.min(a[i], axis=0), np.max(a[i], axis=0)
        a[i] = (a[i] - min_a) / (max_a - min_a + 0.00001)
    return a


data = load_fault_data()
# data = normalize3(data)
# 转换为二维数组
data2d = data.reshape(-1, 9)

# 创建高斯混合模型对象，设置聚类数量为 4
gmm = GaussianMixture(n_components=6)

# 训练模型
gmm.fit(data2d)

# 预测聚类标签
labels = gmm.predict(data2d)

# 将标签转换为三维数组形式
labels3d = labels.reshape(data.shape[0], data.shape[1])

# 输出聚类结果
for i in range(4):
    cluster_data = data[labels3d == i]
    print(f"Cluster {i}: {cluster_data.shape}")