
#基于多源特征协同聚类（无监督）

import torch
import numpy as np
import os
from sklearn.cluster import KMeans

base_dir = '/content/drive/MyDrive/drought_monitor/data_new'
x_path = os.path.join(base_dir, 'dataset_X_2023_new.pt')
y_path = os.path.join(base_dir, 'dataset_Y_2023_new.pt')

print("加载特征并修正波段映射...")
X_tensor = torch.load(x_path) 

# 正确映射逻辑：0-NDVI, 1-VV, 2-VH, 3-VVVH
# 提取最后一个月的特征进行标签构造
last_month = X_tensor[:, -1, :, :, :].cpu().numpy()
ndvi = last_month[:, 0, :, :]
vv = last_month[:, 1, :, :]
vh = last_month[:, 2, :, :]

# 构建有效植被掩膜
valid_mask = ndvi > 0.05 
batch, h, w = ndvi.shape

# 提取特征进行协同聚类
print("执行多源特征协同聚类 (NDVI + VV + VH)...")
features = np.stack([ndvi[valid_mask], vv[valid_mask], vh[valid_mask]], axis=1)

# 聚类划分为 4 个干旱等级
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)

# 语义对齐：根据平均 NDVI 排序，NDVI 越高 = 越不干旱 (等级0)
cluster_means = [features[labels == i, 0].mean() for i in range(n_clusters)]
rank = np.argsort(cluster_means)[::-1] # NDVI 从大到小排列
map_dict = {old: new for new, old in enumerate(rank)}
final_labels = np.array([map_dict[l] for l in labels])

# 重构标签图
Y_tensor = np.zeros((batch, h, w), dtype=np.int64)
Y_tensor[valid_mask] = final_labels

Y_pt = torch.from_numpy(Y_tensor)
print(f"标签生成完毕，分布情况: {torch.bincount(Y_pt.flatten())}")
torch.save(Y_pt, y_path)