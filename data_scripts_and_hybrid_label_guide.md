# 干旱数据脚本与 Hybrid 标签流程说明

本文档用于记录 `zyk_drought_monitor` 项目中与干旱监测、干旱预测、标签构建和预测结果展示相关的核心脚本位置、作用，以及 `data_process_y_hybrid.py` 的详细流程与优势。

---

## 一、核心脚本位置

以下脚本目前都位于项目根目录：

- `zyk_drought_monitor/data_processor.py`
- `zyk_drought_monitor/data_processor_y_threshold.py`
- `zyk_drought_monitor/data_processor_y_threshold_v2.py`
- `zyk_drought_monitor/data_processor_y_kmeans_10ch.py`
- `zyk_drought_monitor/data_process_x_new.py`
- `zyk_drought_monitor/data_process_y_new.py`
- `zyk_drought_monitor/data_process_y_hybrid.py`
- `zyk_drought_monitor/build_forecast_v2_dataset.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_main.py`

---

## 二、脚本用途划分

### 1. 偏干旱监测 / 单月标签生成

这几个脚本更偏向“监测 / 检测”，通常是根据已有时序输入中的最后一个月生成单时相标签。

#### `data_processor.py`
- 作用：从多个月份 TIFF 构建旧版 `X_tensor`
- 输出：通常是 `dataset_X_{year}_new.pt`
- 特点：生成输入特征，不直接生成标签

#### `data_processor_y_threshold.py`
- 作用：基于硬阈值规则生成单月伪标签
- 规则核心：`NDVI + VV + VH`
- 输出：单月 `Y_tensor`
- 更适合：干旱监测 / 单时相分类

#### `data_processor_y_threshold_v2.py`
- 作用：基于增强版阈值规则生成单月伪标签
- 相比 `threshold.py` 增加了：
  - `NDWI` 水体过滤
  - `NDMI / MSAVI` 干旱支撑
- 输出：单月 `Y_tensor`
- 更适合：更稳健的干旱监测标签生成

#### `data_processor_y_kmeans_10ch.py`
- 作用：基于 KMeans 聚类生成单月伪标签
- 使用特征：多通道特征（以 10 通道体系为基础）
- 输出：单月 `Y_tensor`
- 更适合：数据驱动的单月伪标签生成

---

### 2. 偏干旱预测 / 时序标签生成

这几个脚本更偏向 forecasting V2 流程，用于构建时序输入和时序标签。

#### `data_process_x_new.py`
- 作用：构建 forecasting V2 使用的时序输入 `sequence_X`
- 输出：通常是 `sequence_X_{year}.pt`
- 特点：保留完整时间维度

#### `data_process_y_new.py`
- 作用：构建 forecasting V2 使用的时序标签 `sequence_Y`
- 支持两种方式：
  - `threshold`
  - `kmeans`
- 输出：通常是 `sequence_Y_*.pt`
- 特点：按每个月分别生成标签，输出带时间维度

#### `data_process_y_hybrid.py`
- 作用：构建 hybrid 时序伪标签
- 融合方式：
  - 阈值法为主
  - KMeans 为辅修正
- 输出：通常是 `sequence_Y_hybrid_{year}.pt`
- 更适合：希望结合规则法与聚类法的时序预测任务

---

### 3. 预测数据集构建与训练入口

#### `build_forecast_v2_dataset.py`
- 作用：把 `sequence_X` 与 `sequence_Y` 按滑动窗口切成真正用于预测模型训练的样本
- 输入：
  - `sequence_X_{year}.pt`
  - `sequence_Y_threshold_{year}.pt` / `sequence_Y_kmeans_{year}.pt` / `sequence_Y_hybrid_{year}.pt`
- 输出：
  - `forecast_v2_X_{year}.pt`
  - `forecast_v2_Y_*_{year}.pt`
- 作用定位：从“时序原始样本”到“预测训练样本”的桥梁

#### `proposed_attention_optimization/forecast_main.py`
- 作用：读取 `forecast_v2_X / forecast_v2_Y`，训练预测模型
- 可支持的标签方案：
  - `threshold`
  - `kmeans`
  - `hybrid`
- 作用定位：forecasting V2 的训练入口

---

## 三、`data_process_y_hybrid.py` 详细流程说明

这个脚本的目标是：

**针对 `sequence_X_{year}.pt` 的每一个时间步生成标签，并把硬阈值规则与 KMeans 聚类结果融合起来，得到更适合 forecasting 的时序伪标签。**

### 1. 输入与输出

#### 输入
- `sequence_X_{year}.pt`
- 张量形状要求：`(Batch, Time, Channels, H, W)`

#### 输出
- `sequence_Y_hybrid_{year}.pt`
- 张量形状：`(Batch, Time, H, W)`

这意味着它不是只为最后一个月打标签，而是会为整段时间序列中的每个月都生成标签。

---

### 2. 整体流程概览

`data_process_y_hybrid.py` 的流程可以概括为：

1. 读取某一年的 `sequence_X_{year}.pt`
2. 检查输入张量维度和通道是否满足要求
3. 对每一个时间步单独取出一个月的数据 `x_month`
4. 用增强阈值规则生成该月 `threshold` 标签
5. 用 8 维特征做 KMeans 聚类，生成该月 `kmeans` 标签
6. 把 `threshold` 与 `kmeans` 按保守策略融合成 `hybrid` 标签
7. 把所有月份的 `hybrid` 标签堆叠，形成 `Y_sequence`
8. 保存为 `sequence_Y_hybrid_{year}.pt`

---

### 3. 参数与特征设置

脚本内部主要有四类参数：

#### A. 聚类参数
- `N_CLUSTERS = 4`
- `RANDOM_STATE = 42`
- `N_INIT = 10`

作用：控制 KMeans 聚类的类别数和稳定性。

#### B. 有效像元筛选参数
- `VALID_NDVI_THRESHOLD = 0.05`

作用：过滤掉低 NDVI 或无意义区域，避免无效像元参与标签构建。

#### C. 基础干旱阈值
- `NDVI_LIGHT_THRESHOLD`
- `NDVI_MODERATE_THRESHOLD`
- `NDVI_SEVERE_THRESHOLD`
- `VV_HIGH_THRESHOLD`
- `VV_MID_THRESHOLD`
- `VH_HIGH_THRESHOLD`
- `VH_MID_THRESHOLD`

作用：定义轻旱、中旱、重旱的基础规则。

#### D. 辅助门控参数
- `NDWI_WATER_MASK_THRESHOLD`
- `NDMI_DRY_SUPPORT_THRESHOLD`
- `MSAVI_DRY_SUPPORT_THRESHOLD`

作用：
- 用 `NDWI` 排除水体/湿区
- 用 `NDMI / MSAVI` 支持中旱和重旱判断
- 降低噪声和误判

---

### 4. 阈值法部分：`build_threshold_components()`

该函数负责生成每个月的基础规则标签，其思路继承了 `data_processor_y_threshold_v2.py`，但扩展到了时序任务中。

#### 核心步骤

##### 第一步：提取关键通道
提取：
- `NDVI`
- `VV`
- `VH`
- `NDWI`
- `NDMI`
- `MSAVI`

##### 第二步：构造有效区域 `valid_mask`
要求：
- `NDVI / VV / VH` 为有限值
- `NDVI > 0.05`

##### 第三步：水体过滤 `non_water_mask`
规则：
- `NDWI < 0.20`

含义：高 `NDWI` 区域可能是水体或湿区，不参与干旱判断。

##### 第四步：干旱支撑条件
- `dry_support_mask`：`NDMI` 偏低或 `MSAVI` 偏低
- `severe_support_mask`：`NDMI` 偏低且 `MSAVI` 偏低

含义：中旱和重旱不能只看单通道异常，还需要额外证据支持。

##### 第五步：生成阈值标签
标签定义：
- `0`：无旱
- `1`：轻旱
- `2`：中旱
- `3`：重旱

其中：
- `light_mask` 规则最宽松
- `moderate_mask` 更严格
- `severe_mask` 最严格

最后按从轻到重依次赋值，使重旱能够覆盖轻旱和中旱。

---

### 5. 聚类法部分：`generate_kmeans_labels_for_month()`

该函数负责对每个月单独生成聚类标签。

#### 使用的特征
- `NDVI`
- `EVI`
- `NDMI`
- `NDWI`
- `MSAVI`
- `VV`
- `VH`
- `RVI`

相比只用 `NDVI + VV + VH` 的做法，这里用了更多特征，信息更丰富。

#### 核心步骤

##### 第一步：筛选有效像元
要求：
- `NDVI > 0.05`
- 参与聚类的全部 8 个特征都是有限值

##### 第二步：提取多维特征向量
把每个有效像元表示成一个 8 维向量。

##### 第三步：标准化
用 `StandardScaler` 对特征标准化。

作用：避免不同量纲的特征对聚类结果产生不公平影响。

##### 第四步：执行 KMeans
用 `KMeans(n_clusters=4)` 把像元划分为 4 类。

##### 第五步：按 NDVI 均值重排类别编号
KMeans 原始簇编号没有实际物理意义，所以脚本会：
- 计算每个簇的 `NDVI` 平均值
- 按 `NDVI` 从高到低排序
- 重新映射为 `0~3`

语义解释变成：
- NDVI 越高，越接近 `0`（无旱）
- NDVI 越低，越接近 `3`（重旱）

这样聚类结果就能和阈值法的类别体系对齐。

---

### 6. 融合部分：`fuse_labels_for_month()`

这是整个脚本最关键的部分。

它的总原则不是“平均”或“投票”，而是：

**以阈值法为主，以 KMeans 为辅，主要做谨慎升级，不轻易降级。**

#### 具体融合规则

##### 规则 1：两者一致时，直接采用
说明阈值法和聚类法意见一致，标签可信度较高。

##### 规则 2：阈值法判为 `0`，但聚类显示较明显干旱时，可升级
要求：
- 阈值标签为 `0`
- 聚类标签至少达到 `2`
- 并且 `dry_support_mask` 成立

含义：规则法没有抓住异常，但数据分布和辅助特征都在提示干旱，可以提升标签等级。

##### 规则 3：阈值法判为 `1`，聚类判得更重时，可升级
要求：
- 阈值标签为 `1`
- 聚类标签至少达到 `2`
- 同时有干旱支撑

##### 规则 4：阈值法判为 `2`，聚类判为 `3`，且强支撑成立，则升为 `3`
要求：
- 阈值法已经认为存在较严重旱情
- 聚类法进一步认为是重旱
- `severe_support_mask` 成立

##### 规则 5：阈值法已判为 `3` 时，直接保留
防止聚类把高置信重旱错误降级。

##### 规则 6：无效区、水体区、辅助信息缺失区统一置 `0`
保持和项目原有脚本风格一致。

---

### 7. 逐月时序化：`generate_sequence_hybrid_labels()`

这是该脚本区别于单月脚本的核心。

它会：
- 遍历 `Time` 维度
- 每个月都单独调用一次 hybrid 标签构建流程
- 得到每个月的 `y_hybrid`
- 最后沿时间维堆叠

输出：
- `Y_sequence`，形状为 `(Batch, Time, H, W)`

这使它非常适合 forecasting 任务，因为预测模型需要的是“带时间维度的监督信号”。

---

### 8. 运行时的调试信息

脚本在每个月处理时，会打印：
- `threshold` 标签分布
- `kmeans` 标签分布
- `hybrid` 标签分布

作用：
- 观察某个月是否出现类别极端偏斜
- 判断 hybrid 是否过于保守或过于激进
- 辅助后续调参

---

## 四、`data_process_y_hybrid.py` 相对其他脚本的优势

### 1. 相对 `data_processor_y_threshold.py` 的优势

`data_processor_y_threshold.py`：
- 只看最后一个月
- 只用简单阈值
- 输出单月标签 `(Batch, H, W)`

`data_process_y_hybrid.py`：
- 为所有月份生成标签
- 结合规则与聚类
- 输出时序标签 `(Batch, Time, H, W)`

优势总结：
- 更适合预测任务
- 信息利用更充分
- 标签边界更灵活

---

### 2. 相对 `data_processor_y_threshold_v2.py` 的优势

`data_processor_y_threshold_v2.py`：
- 已加入 `NDWI / NDMI / MSAVI`
- 规则更稳健
- 但仍然是单月标签
- 本质上仍是纯规则法

`data_process_y_hybrid.py`：
- 继承了 `threshold_v2` 的稳健思路
- 再引入 KMeans 进行修正
- 扩展为逐月时序标签

优势总结：
- 保留了增强版阈值法的可解释性
- 弥补纯规则法对复杂分布适应不足的问题
- 更适合 forecasting V2 训练链路

---

### 3. 相对 `data_processor_y_kmeans_10ch.py` 的优势

`data_processor_y_kmeans_10ch.py`：
- 只看最后一个月
- 主要依赖聚类结果
- 可解释性较弱

`data_process_y_hybrid.py`：
- 聚类只是辅助
- 阈值法仍负责语义锚定
- 并且按月输出时序标签

优势总结：
- 不会完全被聚类结果牵着走
- 语义更稳定
- 对异常月份、异常聚类分布更鲁棒

---

### 4. 相对 `data_process_y_new.py` 的优势

`data_process_y_new.py`：
- 是时序标签脚本
- 但 `threshold` 与 `kmeans` 仍是二选一

`data_process_y_hybrid.py`：
- 不是二选一
- 而是把两者真正融合成一个新的标签方案

优势总结：
- 更适合伪标签任务
- 避免单一方法的偏差
- 更符合“规则法 + 数据驱动法结合”的实验目标

---

## 五、这个脚本在项目中的位置

可以把 `data_process_y_hybrid.py` 理解为：

**位于“原始时序输入构建”和“预测模型训练”之间的关键桥梁。**

它的上下游关系如下：

1. `data_process_x_new.py`
   - 构建 `sequence_X_{year}.pt`

2. `data_process_y_hybrid.py`
   - 根据 `sequence_X` 构建 `sequence_Y_hybrid_{year}.pt`

3. `build_forecast_v2_dataset.py`
   - 把 `sequence_X / sequence_Y_hybrid` 切成 `forecast_v2_X / forecast_v2_Y_hybrid`

4. `proposed_attention_optimization/forecast_main.py`
   - 读取 `forecast_v2_X / forecast_v2_Y_hybrid` 训练模型

因此，这个脚本不是孤立的数据处理脚本，而是 forecasting V2 训练管线中的核心标签构建模块。

---

## 六、预测结果展示：是怎么实现的

你这个项目里“预测结果展示”不是放在数据处理脚本里完成的，而是集中放在：

- `zyk_drought_monitor/proposed_attention_optimization/forecast_compare.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_spatial_compare.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_dynamic_gif.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_visual_compare_611.py`
- `zyk_drought_monitor/proposed_attention_optimization/export_forecast_tif.py`

这些脚本的基本实现链路是：

1. 读取 `forecast_v2_X_{year}.pt` 和 `forecast_v2_Y_{year}.pt`
2. 取前若干个月作为输入窗口
3. 加载训练好的预测模型 checkpoint
4. 对样本前向推理得到像元级旱情分类结果
5. 把结果渲染成 PNG / 对比图 / GIF / GeoTIFF

---

## 七、简单建议

- 如果你要做“干旱监测 / 干旱检测 / 单月分类”，优先看：
  - `data_processor_y_threshold.py`
  - `data_processor_y_threshold_v2.py`
  - `data_processor_y_kmeans_10ch.py`

- 如果你要做“干旱预测 / forecasting / 时序建模”，优先看：
  - `data_process_x_new.py`
  - `data_process_y_new.py`
  - `data_process_y_hybrid.py`
  - `build_forecast_v2_dataset.py`
  - `proposed_attention_optimization/forecast_main.py`

- 如果你要专门讲“预测结果是怎么展示出来的”，优先看：
  - `proposed_attention_optimization/forecast_compare.py`
  - `proposed_attention_optimization/forecast_spatial_compare.py`
  - `proposed_attention_optimization/forecast_dynamic_gif.py`
  - `proposed_attention_optimization/forecast_visual_compare_611.py`

---

## 八、备注

目前这些脚本已经恢复到原始平铺位置，没有再拆分到新的子文件夹中。
