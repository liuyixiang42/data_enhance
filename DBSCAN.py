import numpy as np
from sklearn.cluster import DBSCAN
from load_fault_data import *

data = load_fault_data()

# 调用 DBSCAN 算法进行聚类
dbscan = DBSCAN(eps=0.2, min_samples=2).fit(data)

# 输出聚类结果
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('聚类的类别数为:', n_clusters_)
print('聚类结果为:', labels)