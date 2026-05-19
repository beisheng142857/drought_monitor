# 数据处理脚本位置说明

本文档用于记录 `zyk_drought_monitor` 项目中与干旱监测/预测相关的数据处理脚本位置，以及它们的大致用途，方便后续查找。

## 脚本位置

以下脚本目前都位于项目根目录：

- `zyk_drought_monitor/data_processor.py`
- `zyk_drought_monitor/data_processor_y_threshold.py`
- `zyk_drought_monitor/data_processor_y_threshold_v2.py`
- `zyk_drought_monitor/data_processor_y_kmeans_10ch.py`
- `zyk_drought_monitor/data_process_x_new.py`
- `zyk_drought_monitor/data_process_y_new.py`
- `zyk_drought_monitor/data_process_y_hybrid.py`

## 用途划分

### 一、偏干旱监测 / 单月标签生成

这几个脚本更偏向“监测/检测”，通常是根据已有时序输入中的最后一个月，生成单时相标签。

#### 1. `data_processor.py`
- 作用：从多个月份 TIFF 构建旧版 `X_tensor`
- 输出：通常是 `dataset_X_{year}_new.pt`
- 特点：生成的是输入特征，不直接生成标签

#### 2. `data_processor_y_threshold.py`
- 作用：基于硬阈值规则生成单月伪标签
- 规则核心：`NDVI + VV + VH`
- 输出：单月 `Y_tensor`
- 更适合：干旱监测 / 单时相分类

#### 3. `data_processor_y_threshold_v2.py`
- 作用：基于增强版阈值规则生成单月伪标签
- 相比 `threshold.py` 增加了：
  - `NDWI` 水体过滤
  - `NDMI` / `MSAVI` 干旱支撑
- 输出：单月 `Y_tensor`
- 更适合：更稳健的干旱监测标签生成

#### 4. `data_processor_y_kmeans_10ch.py`
- 作用：基于 KMeans 聚类生成单月伪标签
- 使用特征：多通道特征（以 10 通道体系为基础）
- 输出：单月 `Y_tensor`
- 更适合：数据驱动的单月伪标签生成

---

### 二、偏干旱预测 / 时序标签生成

这几个脚本更偏向 forecasting V2 流程，用于构建时序输入和时序标签。

#### 5. `data_process_x_new.py`
- 作用：构建 forecasting V2 使用的时序输入 `sequence_X`
- 输出：通常是 `sequence_X_{year}.pt`
- 特点：保留完整时间维度

#### 6. `data_process_y_new.py`
- 作用：构建 forecasting V2 使用的时序标签 `sequence_Y`
- 支持两种方式：
  - `threshold`
  - `kmeans`
- 输出：通常是 `sequence_Y_*.pt`
- 特点：按每个月分别生成标签，输出带时间维度

#### 7. `data_process_y_hybrid.py`
- 作用：构建 hybrid 时序伪标签
- 融合方式：
  - 阈值法为主
  - KMeans 为辅修正
- 输出：通常是 `sequence_Y_hybrid_{year}.pt`
- 更适合：希望结合规则法与聚类法的时序预测任务

---

## 三、预测结果展示：是怎么实现的

你这个项目里“预测结果展示”不是放在数据处理脚本里做的，而是集中放在：

- `zyk_drought_monitor/proposed_attention_optimization/forecast_compare.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_spatial_compare.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_dynamic_gif.py`
- `zyk_drought_monitor/proposed_attention_optimization/forecast_visual_compare_611.py`
- `zyk_drought_monitor/proposed_attention_optimization/export_forecast_tif.py`

这一组脚本的实现逻辑，本质上都是同一条链路：

1. 先读取 `forecast_v2_X_{year}.pt` 和 `forecast_v2_Y_{year}.pt`
2. 从中取前 4 个月作为输入窗口（`FORECAST_INPUT_STEPS = 4`）
3. 加载训练好的预测模型 checkpoint
4. 对每个样本做前向推理，得到每个像元的旱情类别
5. 再把预测类别图转成 PNG / 对比图 / GIF 等可视化结果

也就是说，你的“预测结果展示”本质上不是直接展示网络输出张量，而是把输出的类别图经过颜色映射、标题、颜色条、统计信息等包装后，保存成论文/汇报可以直接使用的图片。

### 1. 模型预测图是怎么生成的

几个脚本内部都用了相似的流程：

- 根据 checkpoint 文件名判断模型类型：
  - `convlstm_attn`
  - `convlstm_no_attn`
  - `convgru`
  - `traj_gru`
- 根据模型类型动态构建网络
- 自动把模型输入通道数调整成当前数据真实通道数
- 对单个样本执行预测
- 通过 `argmax` 得到最终像元类别图

类别图中用的是 4 类干旱等级：

- `无旱`
- `轻旱`
- `中旱`
- `重/特旱`

显示时又统一映射成固定颜色：

- 绿色：无旱
- 浅黄：轻旱
- 橙色：中旱
- 深红：重/特旱

所以最后看到的预测结果图，本质上是一个二维类别栅格，只是被渲染成了彩色旱情分布图。

### 2. 不同展示脚本分别在做什么

#### A. `forecast_compare.py`

这是“多模型总体效果对比”的脚本，不是画单张预测图，而是做整批测试集评估，然后生成总览结果。

它主要输出三类结果：

- 文本总结：每个模型的 `Loss / Accuracy / Macro-F1 / Weighted-F1`
- 指标柱状图：不同模型的精度对比
- 混淆矩阵图：看每一类旱情的混淆情况

所以它展示的是“模型整体预测效果”，更偏定量评估。

默认保存位置是：

- `zyk_drought_monitor/results/forecast_compare_V2/V2_1/forecast_model_summary.txt`
- `zyk_drought_monitor/results/forecast_compare_V2/V2_1/forecast_metrics_comparison.png`
- `zyk_drought_monitor/results/forecast_compare_V2/V2_1/forecast_confusion_matrices_normalized.png`

如果运行时手动改了 `--output_dir`，结果就会保存到你指定的目录。

#### B. `forecast_spatial_compare.py`

这是“同一样本下，不同模型空间预测结果对比”的脚本。

它会把一行图拼出来，通常包含：

- 参考底图（一般是输入窗口最后一个月的 `NDVI`）
- 真实未来旱情图
- 各个模型各自的预测图

这样你可以非常直观地看：

- 真实图斑长什么样
- `ConvLSTM + Attention` 预测成什么样
- `ConvLSTM 无 Attention` 预测成什么样
- `ConvGRU` / `TrajGRU` 又预测成什么样

这个脚本展示的是“空间分布层面谁预测得更像真值”。

默认保存位置是：

- `zyk_drought_monitor/results/forecast_compare_V2/5_10/forecast_spatial_prediction_compare.png`

如果运行时传入 `--output_path`，则保存到对应路径。

#### C. `forecast_dynamic_gif.py`

这个脚本是把连续样本做成动态演变 GIF，用来展示预测结果随时间/样本推进时的变化。

它的做法是：

- 先为每一帧生成一张 PNG
- 每张图左边放参考底图，右边放干旱等级图
- 标题里附带年份、帧号、类别占比信息
- 最后把所有帧合成为一个 GIF

这个脚本有两个模式：

- `gt`：展示真实标签序列
- `pred`：展示模型预测序列

因此它既可以做“真实旱情动态演变”，也可以做“预测旱情动态演变”。

代码里的默认输出位置是：

- 帧目录：`zyk_drought_monitor/drought_outputs/dynamic_forecast/frames/`
- GIF 文件：`zyk_drought_monitor/drought_outputs/dynamic_forecast/drought_dynamic_evolution_forecast.gif`

当前项目里已经实际存在的一组动态预测输出位于：

- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/drought_dynamic_evolution_pred.gif`

这说明你之前至少有一次运行时，把动态预测结果输出到了 `dynamic_pred` 目录，而不是脚本默认的 `dynamic_forecast`。

#### E. `export_forecast_tif.py`

这个脚本是我这次补上的，作用是把某一个预测样本重新导出为带地理参考信息的 `GeoTIFF`。

它的实现思路是：

- 重新读取原始月份 TIFF，按与你构建 `forecast_v2_X_{year}.pt` 时一致的 `patch_size / stride / nodata_threshold` 枚举有效窗口
- 用 `sample_index` 找回这个样本在原始大图中的空间窗口位置
- 加载对应预测模型 checkpoint
- 对该样本做预测，生成 `128 × 128` 的旱情类别图
- 继承原始 TIFF 的 `crs / transform / 分辨率` 等元信息
- 最终写出单波段分类 `GeoTIFF`

它支持导出三类文件：

- 预测分类图 `*_pred.tif`
- 真值分类图 `*_gt.tif`（可选）
- 参考底图 `*_reference_ndvi.tif`（可选）

默认输出位置：

- `zyk_drought_monitor/drought_outputs/prediction_tifs/`

如果你想把预测结果拿到 ArcGIS / QGIS 里继续叠加分析，这个脚本就是最直接可用的版本。

一个典型调用示例是：

```bash
python /root/autodl-tmp/zyk_drought_monitor/proposed_attention_optimization/export_forecast_tif.py \
  --year 2025 \
  --sample_index 0 \
  --checkpoint /root/autodl-tmp/zyk_drought_monitor/data_V2/V2_1/drought_forecast_convlstm_attn_best_threshold_proposed.pth \
  --export_gt \
  --export_reference_band
```

---

## 四、预测结果文件具体放置在哪里

目前从项目目录里能直接看到、与展示相关的结果文件主要在两个地方。

### 1. 动态展示结果目录

目录：`zyk_drought_monitor/drought_outputs/`

目前已存在：

- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/frame_000.png`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/frame_001.png`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/frame_002.png`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/frame_003.png`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/frame_004.png`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/frames/frame_005.png`
- `zyk_drought_monitor/drought_outputs/dynamic_pred/drought_dynamic_evolution_pred.gif`

这里存的是“逐帧 PNG + 合成后的预测动态 GIF”。

另外同目录下还有一批真实监测结果：

- `drought_dynamic_evolution_real.gif`
- `drought_dynamic_evolution_real_1.gif`
- `drought_dynamic_evolution_real_2.gif`
- `drought_level_map_real.png`
- `drought_level_map_comparison_1.png`
- `drought_level_map_comparison_2.png`

它们可以作为预测结果展示时的对照材料。

### 2. 评估与对比结果目录

脚本默认会把更规范的预测评估结果放到：

- `zyk_drought_monitor/results/forecast_compare_V2/`

其中常见子路径包括：

- `zyk_drought_monitor/results/forecast_compare_V2/V2_1/`
- `zyk_drought_monitor/results/forecast_compare_V2/5_10/`

虽然当前目录树里还没直接列出这些文件，但从脚本默认参数来看：

- `forecast_compare.py` 会写到 `forecast_compare_V2/V2_1/`
- `forecast_spatial_compare.py` 会写到 `forecast_compare_V2/5_10/`
- `forecast_visual_compare_611.py` 会写到 `forecast_compare_V2/`

也就是说，`results/forecast_compare_V2/` 才是你这套“预测结果展示”更标准、论文化的结果落盘位置；
而 `drought_outputs/` 更像是偏业务图层、动态图、快速查看结果的输出目录。

---

## 五、可以怎么理解这套展示结构

如果从用途来分，你现在的“预测结果展示”大致是三层：

### 第一层：整体性能展示
- 脚本：`forecast_compare.py`
- 作用：展示模型整体指标好不好
- 结果位置：`results/forecast_compare_V2/...`

### 第二层：空间结果展示
- 脚本：`forecast_spatial_compare.py`
- 作用：展示真实图和预测图在空间分布上像不像
- 结果位置：`results/forecast_compare_V2/...`

### 第三层：动态与解释性展示
- 脚本：
  - `forecast_dynamic_gif.py`
  - `forecast_visual_compare_611.py`
- 作用：
  - 一个看连续演变
  - 一个看错分位置和注意力机制
- 结果位置：
  - `drought_outputs/dynamic_pred/` 或默认的 `drought_outputs/dynamic_forecast/`
  - `results/forecast_compare_V2/...`

所以总结一句话：

你的预测结果展示，不是只靠一张图完成的，而是通过“指标图 + 空间分布图 + 动态 GIF + 错分/注意力解释图”这一整套组合来实现的。

## 简单建议

- 如果你要做“干旱监测 / 干旱检测 / 单月分类”，优先看：
  - `data_processor_y_threshold.py`
  - `data_processor_y_threshold_v2.py`
  - `data_processor_y_kmeans_10ch.py`

- 如果你要做“干旱预测 / forecasting / 时序建模”，优先看：
  - `data_process_x_new.py`
  - `data_process_y_new.py`
  - `data_process_y_hybrid.py`

- 如果你要专门讲“预测结果是怎么展示出来的”，优先看：
  - `proposed_attention_optimization/forecast_compare.py`
  - `proposed_attention_optimization/forecast_spatial_compare.py`
  - `proposed_attention_optimization/forecast_dynamic_gif.py`
  - `proposed_attention_optimization/forecast_visual_compare_611.py`

## 备注

目前这些脚本已经恢复到原始平铺位置，没有再拆分到新的子文件夹中。
