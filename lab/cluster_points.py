import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import random
from matplotlib import font_manager

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']  # 优先使用的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以便结果可重现
np.random.seed(42)

# 定义目标框的位置 [x_min, y_min, x_max, y_max]
bbox = [300, 400, 500, 500]  # [x_min, y_min, x_max, y_max]

# 模拟VLM输出的32个预测点击坐标
n_votes = 32
vlm_outputs = []

# 生成模拟的VLM预测坐标
for _ in range(n_votes):
    # 70%的点落在框内或接近框
    if random.random() < 0.7:
        x = random.uniform(bbox[0] - 20, bbox[2] + 20)
        y = random.uniform(bbox[1] - 20, bbox[3] + 20)
    else:
        # 30%的点是噪声
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
    
    vlm_outputs.append([x, y])

# 转换为numpy数组
vlm_outputs = np.array(vlm_outputs)

# 使用DBSCAN聚类算法找出最大簇
# eps: 两点间的最大距离，min_samples: 形成核心点的最小样本数
dbscan = DBSCAN(eps=50, min_samples=3)
cluster_labels = dbscan.fit_predict(vlm_outputs)

# 找出最大的簇
unique_labels = np.unique(cluster_labels)
largest_cluster_label = -1
largest_cluster_size = 0

for label in unique_labels:
    if label == -1:  # 跳过噪声点
        continue
    cluster_size = np.sum(cluster_labels == label)
    if cluster_size > largest_cluster_size:
        largest_cluster_size = cluster_size
        largest_cluster_label = label

# 获取最大簇中的点并计算中心
if largest_cluster_label != -1:
    largest_cluster_points = vlm_outputs[cluster_labels == largest_cluster_label]
    recommended_click = np.mean(largest_cluster_points, axis=0)
    print(f"推荐点击位置: ({recommended_click[0]:.1f}, {recommended_click[1]:.1f})")
else:
    recommended_click = None
    print("没有找到有效的聚类")

# 可视化结果
plt.figure(figsize=(8, 8))

# 创建点的颜色标记：绿色=最大簇(可能正确)，红色=其他点(可能错误)
colors = ['red' if label != largest_cluster_label else 'green' for label in cluster_labels]
sizes = [80 if label != -1 else 40 for label in cluster_labels]  # 噪声点(-1)设置小一点

# 绘制所有VLM预测点
plt.scatter(vlm_outputs[:, 0], vlm_outputs[:, 1], c=colors, s=sizes)

# 绘制目标框
plt.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]], 
         [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]], 
         'b-', linewidth=2, label=f'目标框 {bbox}')

# 添加标签和图例
plt.title('点击目标预测 (绿色=可能正确, 红色=可能错误)')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('simple_prediction.png', dpi=200)
plt.show() 