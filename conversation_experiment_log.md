# 对话与实验记录（Cursor × AutoDL）

> 目的：防止聊天记录丢失，把关键结论、命令、结果和下一步计划沉淀到项目内。
> 
> 维护约定：从本次开始，我会在每次对话后追加“最新更新”小节。

---

## 项目背景
- 项目：`zyk_drought_monitor`
- 目标：干旱监测 + 未来干旱预测
- 当前重点：对比 ConvLSTM（无 attention）与 ConvLSTM（有 attention）模型效果

## 已完成事项（截至本次）
1. 新增评估脚本：`evaluate_checkpoints.py`
   - 可批量评估多个 `.pth` 模型
   - 指标：`Loss`、`Accuracy`、`Macro-F1`、`Weighted-F1`
   - 输出：`classification_report` 与 `confusion_matrix`

2. 新增可视化脚本：`visualize_model_compare.py`
   - 生成模型指标对比柱状图
   - 生成混淆矩阵（行归一化）对比图
   - 已修复中文字体显示（支持 `SimHei.ttf`）
   - 已修复右侧子图/色条布局问题

## 关键评估结果（2024 测试集）
1. `drought_convlstm_best_2021_2024_413.pth`（attention）
   - Acc: **0.9813**
   - Macro-F1: **0.9685**
   - Weighted-F1: **0.9814**

2. `drought_convlstm_best_2021_2024_20260414_211714.pth`（attention）
   - Acc: **0.9760**
   - Macro-F1: **0.9570**
   - Weighted-F1: **0.9759**

3. `drought_convlstm_best.pth`（无 attention）
   - Acc: **0.8235**
   - Macro-F1: **0.7722**
   - Weighted-F1: **0.8505**

结论：attention 版本显著优于初版无 attention；当前最佳为 `..._413.pth`。

## 结果文件位置
- 文本结果：`results/model_compare/*_classification_report.txt`
- 混淆矩阵：`results/model_compare/*_confusion_matrix.pt`
- 可视化图：`results/model_compare/figures/`

---

## 最新更新
### 2026-04-17
- 用户要求建立长期保存文档，避免会话记录丢失。
- 已创建本文件：`/root/autodl-tmp/zyk_drought_monitor/conversation_experiment_log.md`
- 约定：后续每次对话后，追加关键变更与结论。

### 2026-04-17（数据链路复盘）
- 复盘了数据获取与预处理链路：`gee_downloader.py`、`data_processor.py`、`data_processor_y.py`。
- 识别的优化重点：
  1) `gee_downloader.py` 中 ROI 通过 `roi.getInfo()` 拉回本地，建议改为直接传 `ee.Geometry`，减少客户端阻塞；
  2) Sentinel-1 仅按 DESCENDING 回退逻辑可再增强（建议统一升降轨或做月内轨道统计）；
  3) `data_processor.py` 当前一次性 `np.stack([ds.read() ...])` 占内存，建议窗口化读取并流式写出；
  4) `data_processor_y.py` 通道假设与实际特征定义可能不一致（当前注释写 NDWI，但下载链路是 NDVI/VV/VH/VVVH），需统一标签规则输入通道；
  5) 路径存在 Colab 与 AutoDL 混用，建议参数化与命令行化，统一可复现入口。
- 下一步建议：优先做“路径参数化 + 通道定义统一 + 流式切片”三件事，再迭代标签构建策略。

### 2026-04-17（GEE分辨率评估）
- 用户询问 `gee_downloader.py` 的数据精度是否偏低。
- 已确认当前导出参数为 `scale=100`，相较于 Sentinel-2/Sentinel-1 常用 10m 分辨率明显更粗。
- 结论：用于区域级趋势监测可用，但对田块级/细碎地物识别会损失细节。
- 建议：可做 100m vs 30m/20m 对比实验（同一区域同月份），用 Macro-F1（尤其重旱类）与推理耗时综合决策。

### 2026-04-17（Colab→AutoDL 数据迁移方案）
- 用户提出 Colab token 不足，需在 AutoDL 使用 Google Drive 上的原始数据并继续训练。
- 建议主方案：使用 `rclone` 将 Drive 目录同步到 AutoDL 本地目录，再在 AutoDL 运行 `data_processor.py` / `data_processor_y.py` / 训练脚本。
- 补充建议：先做小规模验证（1-2个月 TIFF），确认路径与维度后再全量同步。

### 2026-04-17（rclone 安装卡住与 Google 登录报错）
- 用户反馈 `curl https://rclone.org/install.sh | sudo bash` 看似卡住，以及 Google OAuth 页面报 `400 invalid_request`。
- 解释：
  1) 在 `root` 环境下使用 `sudo` 可能导致脚本执行体验异常（等待/静默）；
  2) `invalid_request` 常见于 rclone 版本较旧、默认 OAuth 流程受限或客户端校验失败。
- 建议：
  1) 直接使用无 `sudo` 安装（或 `apt install rclone`）并确认版本；
  2) 使用最新 rclone + `rclone authorize "drive"` 方式获取 token；
  3) 若仍失败，改用“自建 Google OAuth Client ID/Secret”配置 rclone。

### 2026-04-17（rclone 版本确认）
- 用户当前版本：`rclone v1.53.3-DEV`，并提示 `rclone.conf` 尚未创建。
- 结论：该版本偏旧且为 DEV 构建，不建议继续用于 Google Drive OAuth。
- 建议升级到稳定新版本（建议 `v1.66+`），然后重新执行 `rclone config`。
- 说明：`Configuration file doesn't exist` 在首次配置前是正常现象。

### 2026-04-17（执行清单下发）
- 用户确认需要“可直接复制”的 rclone 升级与配置命令清单。
- 已提供：卸载旧版/安装新版/Drive 授权/测试连接/同步数据/校验文件的完整步骤。

### 2026-04-17（rclone 异常输出解读）
- 用户反馈 `apt` 提示“rclone 已是最新版本”，但执行 `rclone` 仍报 `command not found`。
- 结论：属于“包状态与可执行文件不一致”（可能是二进制被手动删除、PATH 未命中或安装损坏）。
- 建议：先 `apt install --reinstall -y rclone`，再检查 `dpkg -L rclone` 与 `/usr/bin/rclone` 是否存在；必要时用官方二进制手动安装到 `/usr/local/bin/rclone`。

### 2026-04-17（rclone 版本仍为 1.53.3）
- 用户确认 `rclone` 命令已恢复，但版本仍是 `v1.53.3-DEV`。
- 说明：Ubuntu 仓库通常提供旧版本，`apt install` 不会升级到 rclone 最新稳定版。
- 建议：改用官方二进制手动安装到 `/usr/local/bin/rclone`，并用 `type -a rclone` 确认命令优先级。

### 2026-04-17（官方包下载过慢）
- 用户反馈从 `downloads.rclone.org` 下载 `rclone-current-linux-amd64.zip` 速度很慢。
- 建议临时策略：
  1) 使用续传下载（`wget -c` 或 `curl -C -`）避免重复；
  2) 改用 GitHub Release 直链下载；
  3) 若只为迁移 Drive 数据，先用现有版本完成配置（必要时结合自建 OAuth）。

### 2026-04-17（OAuth invalid_request + GitHub 404）
- 用户再次遇到 `rclone config` 授权页面 `400 invalid_request`。
- 结论：默认 `rclone` 公共 OAuth 客户端对该账号/环境不可用，需改用“自建 Google OAuth Client”。
- 用户反馈 GitHub 下载 `rclone-current-linux-amd64.zip` 返回 404。
- 解释：新版本发布资产命名为 `rclone-v<版本>-linux-amd64.zip`，`rclone-current-*` 在该发布页不可用。
- 建议：通过 GitHub API 先取版本号，再拼接 `rclone-v${VER}-linux-amd64.zip` 下载。

### 2026-04-17（自建 OAuth 后 token 交换超时）
- 用户使用自建 `client_id/client_secret` 完成授权并拿到 verification code，但在请求 `https://oauth2.googleapis.com/token` 时 `dial tcp ... i/o timeout`。
- 结论：当前是服务器到 Google OAuth 端点的网络连通性问题（非配置项本身错误）。
- 处理建议：
  1) 先轮换/撤销已暴露的 OAuth 密钥；
  2) 在 AutoDL 做连通性测试（`oauth2.googleapis.com`、`www.googleapis.com`）；
  3) 若不通，改用代理；若无法代理，改为“本地机下载 Drive → rsync/scp 上传 AutoDL”的中转方案。

### 2026-04-17（Google 全部超时后的落地方案）
- 用户测试 `accounts.google.com / oauth2.googleapis.com / www.googleapis.com` 均超时，确认 AutoDL 出口网络无法直连 Google。
- 明确：此时服务器端 `rclone + Google` 基本不可行，需改“本地中转”或“反向代理隧道”。
- 已给出两条实操路径：
  1) 本地电脑执行 `rclone copy gdrive -> autodl(sftp)` 直传，不走阿里云盘中转；
  2) 本地有代理时，使用 SSH 反向端口把本地代理映射到 AutoDL，再在服务器上导出 `https_proxy` 使用。

---

## 明日执行手册（详细版）

> 目标：不依赖 AutoDL 访问 Google，把 Google Drive 的 GEE 数据稳定传到 AutoDL，然后继续预处理与训练。

### 0. 安全与准备（先做）
1. 由于之前在终端中暴露过 `client_secret` 与授权码，先在 Google Cloud 控制台执行：
   - 删除或重置旧 OAuth Client Secret
   - 重新生成新的 OAuth 凭据（若仍使用该项目）
2. 确认 AutoDL SSH 连接信息：
   - IP
   - 端口
   - 用户名（通常 `root`）
   - 密码或私钥
3. 在 AutoDL 创建目标目录：
   - `/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs`
   - `/root/autodl-tmp/zyk_drought_monitor/data`

### 1. 本地机器准备（推荐主路径）
> 在你自己的电脑执行（能访问 Google 的环境）

1. 安装 rclone（本地）
2. 本地配置 Google Drive remote（假设名为 `gdrive`）：
   - `rclone config`
   - 新建 remote：`gdrive`
   - storage 选 `drive`
   - 完成网页授权
3. 验证本地可读 Drive：
   - `rclone lsd gdrive:`
   - `rclone lsf gdrive:GEE_Drought_Project`

### 2. 本地配置 AutoDL 的 SFTP remote
1. 本地执行 `rclone config`
2. 新建 remote：`autodl`
3. storage 类型选 `sftp`
4. 依次填写：
   - host: 你的 AutoDL IP
   - user: `root`
   - port: 你的 SSH 端口
   - pass/key: 按你的 SSH 登录方式
5. 验证：
   - `rclone lsd autodl:/root/autodl-tmp/zyk_drought_monitor`

### 3. 数据直传（不走阿里云盘）
#### 3.1 先做小样本验证（强烈建议）
- 执行：
  - `rclone copy gdrive:GEE_Drought_Project/Fused_100m_2023_05.tif autodl:/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs -P`
  - `rclone copy gdrive:GEE_Drought_Project/Fused_100m_2023_06.tif autodl:/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs -P`
- 到 AutoDL 检查：
  - `ls -lh /root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs`

#### 3.2 全量传输
- 执行：
  - `rclone copy gdrive:GEE_Drought_Project autodl:/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs -P --transfers 8 --checkers 16 --retries 10 --low-level-retries 20`

#### 3.3 中断续传
- 直接重复同一条 `rclone copy` 命令即可，rclone 会跳过已完成文件。

### 4. AutoDL 侧校验
在 AutoDL 执行：
1. 文件数量与体积：
   - `ls -lh /root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs`
2. 随机抽查 TIFF 是否可读（可选）
3. 记录已同步月份清单（建议写入日志）

### 5. 继续你的数据预处理
1. 修改/确认 `data_processor.py` 输入目录为：
   - `/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs`
2. 运行 `data_processor.py` 生成 `dataset_X_*.pt`
3. 运行 `data_processor_y.py` 生成 `dataset_Y_*.pt`
4. 检查张量：shape、dtype、类别分布是否正常

### 6. 训练与评估（沿用现有脚本）
1. `main.py` 训练
2. `evaluate_checkpoints.py` 对比模型
3. `visualize_model_compare.py` 可视化

### 7. 备选方案（仅当你想在 AutoDL 直接连 Google）
- 你本地有代理时，可用 SSH 反向端口给 AutoDL 提供代理出口：
  1) 本地开隧道：`ssh -N -R 17890:127.0.0.1:7890 root@<autodl_ip> -p <port>`
  2) AutoDL 设置：
     - `export http_proxy=http://127.0.0.1:17890`
     - `export https_proxy=http://127.0.0.1:17890`
  3) 测试 `curl -I https://accounts.google.com`
  4) 再尝试 `rclone config`

### 8. 明日最短执行顺序（建议照这个走）
1) 本地配置 `gdrive` 与 `autodl` 两个 remote
2) 先传 2 个 TIFF 小样本
3) AutoDL 跑一次 `data_processor.py`/`data_processor_y.py` 验证链路
4) 成功后再全量传输
5) 全量训练 + 评估 + 可视化

### 2026-04-17（rclone 新版本安装成功）
- 终端确认已升级到 `rclone v1.73.4`，下载与解压过程完整成功。
- 当前状态：`rclone` 可正常使用，版本问题已解决；后续阻塞点主要是 AutoDL 到 Google 网络不可达。
- 下一步执行建议：优先采用“本地 rclone 直传到 AutoDL(sftp)”路径，不再依赖服务器直连 Google。

### 2026-04-17（GitHub 上传前检查）
- 用户询问 `run.ipynb` 前几行命令是否适合用于上传。
- 结论：`!python ./zyk_drought_monitor/main.py` 这类运行命令本身可以保留，但该 notebook 已包含大量训练输出，直接上传会导致仓库臃肿且可读性差。
- 建议：上传前清理 notebook 输出，并排除大文件（`.pt`、`.pth`、数据目录、结果图大文件）后再推送。

### 2026-04-17（GitHub push 被拦截原因）
- 用户本地提交已成功，但 `git push origin master` 被 GitHub Push Protection 拒绝（`GH013`）。
- 根因：提交历史中检测到疑似 `GitHub Personal Access Token`（路径指向 `code/chushi.ipynb` 与其 checkpoint，commit `409f500...`）。
- 同时发现本次提交包含大量不应入库文件（`.ipynb_checkpoints`、`drought_outputs` 等）。
- 建议处理：
  1) 立刻撤销/轮换泄露的 token；
  2) 增加 `.gitignore`；
  3) 使用历史重写工具移除敏感内容后再强推（不要点“unblock secret”放行）。

### 2026-04-17（上传策略调整：保留对比结果）
- 用户希望上传 `zyk_drought_monitor`，并保留模型对比结果。
- 已调整 `.gitignore`：默认继续忽略 `results/*`，但放行 `results/model_compare/**`，用于上传分类报告、混淆矩阵与可视化图。
- 下一步提醒：如果这些文件之前被忽略，需要 `git add -f` 或重新 `git add` 以纳入版本控制。

### 2026-04-17（历史清理准备）
- 用户确认继续执行 Git 历史清理，以解决 `GH013` secret push protection 拦截。
- 已准备提供最短命令链：安装 `git-filter-repo`、删除问题文件历史、清理引用并强推。

### 2026-04-18（再次 push 仍被拒）
- 用户通过 notebook/脚本再次执行 GitHub 同步，`git commit` 成功但 `git push` 仍被 `GH013` 拦截。
- 原因：GitHub 检查的是“此次 push 涉及的整段提交历史”，不是只检查最新一次 commit；历史中的 PAT 未被真正移除，所以仍会拒绝。
- 额外现象：当前提交还把 `drought_outputs/*` 加进去了，说明同步脚本/当前 `.gitignore` 规则没有完全阻止结果文件入库。
- 结论：必须先重写历史清除敏感内容，再推送；仅继续 commit 或重新 push 不会解决问题。

### 2026-04-18（git-filter-repo 未安装）
- 用户执行历史清理命令时收到 `git: 'filter-repo' is not a git command`。
- 结论：当前环境未安装 `git-filter-repo`，需要先安装该工具，或者退而使用 BFG / `git filter-branch`。
- 建议优先：`apt install git-filter-repo` 或 `python3 -m pip install git-filter-repo`，安装后再执行历史重写。

### 2026-04-18（filter-repo 路径错误）
- 用户在 `zyk_drought_monitor` 子目录执行 `git filter-repo` 时，传入了 `../code/chushi.ipynb`。
- 报错 `Invalid path component '..'` 的意思是：`git-filter-repo` 只接受仓库根目录内的相对路径，不接受 `..` 这种跳到上级目录的写法。
- 结论：需要先确认真正的 Git 仓库根目录，并在仓库根执行清理命令，路径写成如 `code/chushi.ipynb`。

### 2026-04-18（上传范围修正）
- 用户明确：要上传到 GitHub 的仓库目录是 `zyk_drought_monitor`，不是上级 `autodl-tmp`。
- 同时说明：上级 `code` 文件夹已经移出 `zyk_drought_monitor`，因此后续 Git 操作只围绕 `zyk_drought_monitor` 本身进行。
- 已按新要求调整：允许上传结果文件，包括 `drought_outputs/**` 与 `results/model_compare/**`。
- 建议后续采用“在 `zyk_drought_monitor` 内单独初始化/维护 Git 仓库”的方式，避免再受上级目录历史污染。

### 2026-04-18（独立仓库首次提交与推送被拒）
- 用户已在 `zyk_drought_monitor` 内完成一次独立初始化提交，日志显示为 `root-commit`，说明这是该目录新仓库的第一个本地提交。
- 当前 `git push origin master` 被拒绝为 `fetch first`，原因是远端 GitHub 仓库已存在提交历史，而本地是全新历史，二者互不相干。
- 这种场景下若确认要用本地仓库内容覆盖远端，应使用 `git push origin master --force`；若想保留远端历史，则需要先拉取并合并，但通常不适用于“重建为独立仓库”的情形。

### 2026-04-18（是否保留旧仓库历史的决策）
- 用户担心远端旧仓库可能还有用，不确定是否应直接覆盖。
- 建议采用保守策略：先把当前本地内容推到一个新分支或新仓库做备份，再检查远端旧仓库内容是否需要迁移。
- 推荐顺序：
  1) `git push origin master:cursor-backup` 或推到新仓库；
  2) 在 GitHub 网页查看旧仓库历史与文件；
  3) 确认旧内容无价值后，再 `--force` 覆盖主分支。

### 2026-04-18（提供 notebook 风格上传脚本）
- 用户希望得到一个像 `run.ipynb` 前几行那样、可直接在 notebook/终端执行的完整 GitHub 分支上传代码块。
- 已给出包含目录切换、状态检查、提交与推送到 `cursor-backup` 分支的完整命令模板。

### 2026-04-18（放弃本地 rclone 的替代数据连接方案）
- 用户反馈在本地使用 `rclone` 也无法连接 Google Drive，希望不要再走该方案。
- 已给出替代思路：
  1) 直接从 Google Drive 网页批量下载到本地后，用 SFTP/SSH 上传到 AutoDL；
  2) 在本地把 Drive 文件先打包，再单文件上传到 AutoDL 解压；
  3) 若 GEE 端仍可用，可考虑改为导出到其他更易访问的平台（如本地磁盘、阿里云 OSS、腾讯 COS、Dropbox 等）；
  4) 若数据量不大，也可在本地完成预处理后只上传 `.pt` 成品到 AutoDL。
- 当前推荐：最少折腾的是“本地网页下载原始 TIFF -> 上传 AutoDL”或“本地预处理后只传 `.pt`”。

### 2026-04-18（TIFF/压缩包上传细化）
- 用户要求详细说明如何把原始 TIFF 或打包后的压缩包上传到 AutoDL。
- 已补充说明两条主路径：
  1) 单独上传 TIFF：适合文件不多、想逐月核对；
  2) 先打包再上传：适合文件多、想减少传输管理成本。
- 推荐顺序：先在本地小批量测试上传 1-2 个 TIFF，确认 AutoDL 目标目录与后续预处理脚本可读；再决定是否全量单传或改为压缩包方案。

### 2026-04-18（AutoDL 连接信息确认）
- 用户询问 AutoDL 的 IP、SSH 端口与密码分别看哪里。
- 已说明：
  1) IP 一般在 AutoDL 控制台实例详情/连接信息页查看；
  2) SSH 端口通常就是控制台给出的 SSH 端口，不是项目代码里的 `config.py` 训练配置；
  3) 密码通常是实例连接密码/root 密码，或使用平台提供的 SSH 密钥，而不是 GitHub/Google 等其他密码。

### 2026-04-18（已确认当前 AutoDL SSH 信息）
- 用户提供连接命令：`ssh -p 53144 root@connect.bjb1.seetacloud.com`
- 因此当前可确定：
  1) 主机地址：`connect.bjb1.seetacloud.com`
  2) SSH 端口：`53144`
  3) 用户名：`root`
- 后续 `scp` / `rsync` / SFTP 上传命令都应基于这组三元信息构造。

### 2026-04-18（GEE 扩展到 2015–2020 时的失败原因判断）
- 用户说明：原始主数据集为 2020–2025 年 5–9 月，现尝试扩展下载 2015–2020 年 5–9 月以增加样本量。
- 结合 GEE 任务报错 `Image.normalizedDifference: No band named 'B8'. Available band names: [].`，判断当前失败的直接原因是：某些月份的 Sentinel-2 月合成结果为空，导致无法计算 `NDVI(B8, B4)`。
- 分析结论：
  1) 2015–2016 年出现失败，多半与 Sentinel-2 数据时序覆盖不足/不适配有关；
  2) 2017–2018 年个别月份失败，不代表整年不可用，更可能是月内影像在“日期过滤 + 云量阈值 + QA60 掩膜”之后变为空；
  3) 当前 `gee_downloader.py` 缺少“空集合检查”和“失败月份记录/跳过”机制，因此会把个别缺月放大成批处理失败。
- 风险判断：如果后续建模仍要求每年固定使用 5–9 月 5 个时相，则缺少任意一个月份都会影响固定长度时序输入构建。

### 2026-04-18（农业干旱任务的月份范围建议）
- 用户说明：研究重点是农业干旱，因此只重点关注每年 5–9 月数据。
- 当前建议：暂时不急于加入更多月份，先把 5–9 月数据链路做稳定。
- 原因：
  1) 5–9 月本身与农业生长季和夏季旱情关系最直接；
  2) 额外加入非生长季月份，可能让模型学到更多季节性差异而不一定提升农业干旱识别；
  3) 现阶段主要瓶颈仍是下载完整性、通道定义一致性和标签构建稳定性，而不是月份数量不足。
- 后续扩展建议：若未来要尝试扩充时间窗口，优先考虑做 `5–9 月` 与 `4–9 月` 的对比实验，而不是直接扩展到全年。

### 2026-04-18（会话记录维护约定更新）
- 用户要求：将当前及以后与项目相关的会话持续记录到 `conversation_experiment_log.md` 中。
- 维护约定更新：后续对话若产生明确结论、关键决策、重要命令、代码修改建议或实验安排，应继续追加到该日志文件中，作为长期项目记录。
- 新增执行要求：后续每次与项目相关的对话，都应尽量同步追加到该日志文件，而不只记录部分关键轮次。

### 2026-04-18（GitHub 同步状态辨析）
- 用户在 notebook/终端中执行同步脚本后，误以为“无法上传到分支”。
- 根据终端输出 `master -> cursor-backup` 与 `Your branch is up to date with 'origin/cursor-backup'`，已确认这次推送实际成功。
- 关键解释：当前是“本地分支名仍为 `master`，但其跟踪并推送到远端分支 `cursor-backup`”的状态，因此不是上传失败，而是本地/远端分支名不同导致的理解混淆。

### 2026-04-18（预处理脚本检查与修正）
- 用户确认：当前已下载所有年份的 4 月数据，希望检查并修正 `zyk_drought_monitor` 中的数据预处理脚本。
- 检查结论：项目内原有预处理脚本存在以下主要问题：
  1) `data_processor.py` 与 `data_processor_y.py` 仍保留 Colab/Google Drive 路径，不能直接在当前 AutoDL 项目目录运行；
  2) `data_processor_y.py` 仍把第 2 通道假定为 `NDWI`，与当前 GEE 导出特征 `NDVI/VV/VH/VVVH` 不一致；
  3) `data_process_x_new.py` 原脚本虽然尝试做鲁棒切片，但缺少 `return` 返回张量，主程序 `torch.save(X_tensor, ...)` 实际会保存失败；
  4) 当前项目 `data/` 目录为空，`data_raw/` 目录尚未建立，说明预处理目录规范仍需统一。
- 已完成修正：
  1) 重写 `data_process_x_new.py`，统一到项目本地路径 `/root/autodl-tmp/zyk_drought_monitor/data_raw/gee_tiffs` 和输出目录 `/root/autodl-tmp/zyk_drought_monitor/data`；
  2) 新版 X 脚本加入文件存在性检查、窗口化读取、无效值掩膜、可选按通道归一化，并正确返回 `torch.Tensor`；
  3) 重写 `data_process_y_new.py`，按当前通道定义 `0=NDVI, 1=VV, 2=VH, 3=VVVH` 构建伪标签，其中聚类特征使用 `NDVI + VV + VH`；
  4) 新版 Y 脚本支持按年份读取 `dataset_X_<year>.pt` 并输出 `dataset_Y_<year>.pt`。
- 后续优化建议：
  1) 将 `TARGET_YEAR` 和月份列表参数化，逐年批处理生成 2020–2025（以及后续扩展年份）的 X/Y 数据；
  2) 建立统一原始数据目录 `data_raw/gee_tiffs/`，避免 GEE 导出结果散落；
  3) 在生成完每年 X/Y 后，增加 shape、类别分布和缺失比例检查；
  4) 训练入口 `main.py` 与配置文件中的通道注释仍写有 `NDWI`，后续建议同步清理，避免文档与真实数据定义不一致。

### 2026-04-18（旧版预处理脚本与训练注释同步修正）
- 用户要求：优化 `data_processor.py`、`data_processor_y.py`，但保留原有路径设置不变；同时只修正 `main.py` 与 `config.py` 中的错误注释。
- 已完成处理：
  1) `data_processor.py` 在保留原有 `base_dir` 路径的前提下，加入 TIFF 文件存在性检查、窗口化读取、无效值掩膜、图块过滤与可选归一化，使其比原先一次性 `np.stack(ds.read())` 的方式更稳健；
  2) `data_processor_y.py` 改为与当前真实通道定义一致：`0=NDVI, 1=VV, 2=VH, 3=VVVH`，并用 `NDVI + VV + VH` 的 KMeans 协同聚类生成伪标签；
  3) `config.py` 中 ConvLSTM / TrajGRU 的输入通道注释已从 `NDWI` 修正为 `VV / VH / VVVH` 的真实定义；
  4) `main.py` 本次未改动实际逻辑，相关注释未发现必须修正的错误项。
- 说明：`data_processor.py` 仍可能在编辑器中出现少量类型提示类 diagnostics，但当前未发现会阻止脚本运行的 linter 错误。

### 2026-04-18（预处理输出目录调整为 data_proc）
- 用户要求：将预处理结果保存地址改到项目中的新文件夹 `data_proc`。
- 已完成调整：
  1) `data_processor.py` 现在会在原始 `base_dir` 下自动创建 `data_proc/`，并将 `dataset_X_2023.pt` 保存到该目录；
  2) `data_processor_y.py` 现在会在其 `base_dir` 下自动创建 `data_proc/`，并将 `dataset_Y.pt` 保存到该目录；
  3) 目录创建逻辑已内置，无需手动预先建目录。

