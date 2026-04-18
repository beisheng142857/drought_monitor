# 实验结果记录

> 用途：集中记录不同模型、不同标签方案、不同时间窗口下的训练/验证/测试结果，便于横向对比。
>
> 维护建议：每次训练完成后，追加一条新记录；不要覆盖旧结果。

---

## 记录模板

### 实验名称

- 日期：
- 模型：
- 标签方案：
- 输入时间窗口：
- 训练集：
- 验证集：
- 测试集：
- 权重文件：

### 指标结果

- 最佳训练 Accuracy：
- 最佳训练 Macro-F1：
- 最佳验证 Accuracy：
- 最佳验证 Macro-F1：
- 测试 Accuracy：
- 测试 Macro-F1：

### 其他信息

- 类别权重：
- 训练轮数：
- 备注：

---

## 已记录结果

## 四模型监测对比表（建议优先填写 threshold）


| 模型              | 标签方案      | 最佳验证 Acc           | 最佳验证 Macro-F1      | 测试 Acc              | 测试 Macro-F1         | 权重文件                                          | 备注  |
| --------------- | --------- | ------------------ | ------------------ | ------------------- | ------------------- | --------------------------------------------- | --- |
| ConvLSTM-Attn   | threshold | 0.9493719736735026 | 0.8318818452269964 | 0.9220568339029948  | 0.7507524482185169  | `drought_convlstm_attn_best_threshold.pth`    |     |
| ConvLSTM-NoAttn | threshold | 0.9458802541097006 | 0.8204600026491946 | 0.9274431864420573  | 0.7524283255388613  | `drought_convlstm_no_attn_best_threshold.pth` |     |
| ConvGRU         | threshold | 0.9463617536756728 | 0.8320800416725143 | 0.9198201497395834  | 0.7404333997973215  | `drought_convgru_best_threshold.pth`          |     |
| TrajGRU         | threshold | 0.5648535975703487 | 0.4484782944563748 | 0.39481353759765625 | 0.34140462269811395 | `drought_traj_gru_best_threshold.pth`         |     |
| ConvLSTM-Attn   | kmeans    |                    |                    |                     |                     | `drought_convlstm_attn_best_kmeans.pth`       |     |
| ConvLSTM-NoAttn | kmeans    |                    |                    |                     |                     | `drought_convlstm_no_attn_best_kmeans.pth`    |     |
| ConvGRU         | kmeans    |                    |                    |                     |                     | `drought_convgru_best_kmeans.pth`             |     |
| TrajGRU         | kmeans    |                    |                    |                     |                     | `drought_traj_gru_best_kmeans.pth`            |     |


---

### 2026-04-18｜ConvLSTM + KMeans

- 日期：2026-04-18
- 模型：ConvLSTM
- 标签方案：kmeans
- 输入时间窗口：5–9 月（5 个时相）
- 训练集：2021–2023
- 验证集：2024
- 测试集：2025
- 权重文件：`/root/autodl-tmp/zyk_drought_monitor/data/drought_convlstm_best_kmeans.pth`

### 指标结果

- 最佳训练 Accuracy：0.7064
- 最佳训练 Macro-F1：0.6851
- 最佳验证 Accuracy：0.6381
- 最佳验证 Macro-F1：0.6143
- 测试 Accuracy：0.4782
- 测试 Macro-F1：0.3150

### 其他信息

- 类别权重：`[0.47259548, 1.0349919, 1.4333647, 4.541834]`
- 训练轮数：14
- 备注：作为当前 ConvLSTM 基线，整体效果一般，测试集泛化较弱。

---

### 2026-04-18｜ConvLSTM + Threshold

- 日期：2026-04-18
- 模型：ConvLSTM
- 标签方案：threshold
- 输入时间窗口：5–9 月（5 个时相）
- 训练集：2021–2023
- 验证集：2024
- 测试集：2025
- 权重文件：`/root/autodl-tmp/zyk_drought_monitor/data/drought_convlstm_best_threshold.pth`

### 指标结果

- 最佳训练 Accuracy：0.9473
- 最佳训练 Macro-F1：0.8507
- 最佳验证 Accuracy：0.9497
- 最佳验证 Macro-F1：0.8170
- 测试 Accuracy：0.9044
- 测试 Macro-F1：0.7027

### 其他信息

- 类别权重：`[1.1881925, 0.46059743, 1.0270774, 73.22863]`
- 训练轮数：50
- 备注：当前明显优于 KMeans 标签方案，可作为现阶段主线标签方案；但第 4 类权重极高，后续仍需关注类别分布与混淆矩阵。

