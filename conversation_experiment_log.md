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

### 2026-04-18（拉取 GitHub cursor-backup 分支到项目目录）
- 用户询问：如何把仓库中的 `cursor-backup` 分支内容拉取到本地 `zyk_drought_monitor` 文件夹。
- 说明要点：
  1) 若当前 `zyk_drought_monitor` 已是一个 Git 仓库并已关联 `origin`，可在该目录直接 `git fetch origin` 后切换/拉取 `origin/cursor-backup`；
  2) 若本地当前分支就是 `master` 且跟踪远端 `origin/cursor-backup`，则在该目录执行 `git pull` 本质上就会拉取 `cursor-backup` 的最新内容；
  3) 若只是想把远端分支最新内容覆盖到当前工作目录，需先提交或暂存本地改动，避免拉取时冲突或被拒绝。

### 2026-04-18（新增硬阈值版伪标签脚本）
- 用户要求：在保留现有 KMeans 聚类版 `data_processor_y.py` 的同时，新建一个“硬阈值版”的标签生成脚本，并让输出 `.pt` 文件名不要与现有结果冲突。
- 已完成处理：
  1) 新建脚本：`data_processor_y_threshold.py`；
  2) 该脚本按照当前真实通道定义 `0=NDVI, 1=VV, 2=VH, 3=VVVH`，基于最后一个月特征用硬阈值规则生成 4 类伪标签；
  3) 输出文件名已区分为 `dataset_Y_2022_threshold.pt`，避免与聚类版 `dataset_Y_2022.pt` 混淆；
  4) 脚本 lint 检查通过，无新增报错。

### 2026-04-18（硬阈值版标签的阈值来源说明）
- 用户询问：`data_processor_y_threshold.py` 中 NDVI / VV / VH 的阈值是如何设置的。
- 当前说明：这组阈值属于“工程初始阈值”，主要目标是先构造一套与当前通道定义一致、可直接运行并便于与 KMeans 方案对比的硬阈值基线，并非已经过论文或本地区样本统计严格校准。
- 具体依据：
  1) `NDVI` 阈值（0.60 / 0.40 / 0.20）对应“植被较好 / 一般 / 偏差 / 很差”的经验分层思路；
  2) `VV` / `VH` 阈值采用较常见的 SAR 后向散射经验分界思路，用来辅助区分植被/土壤湿度偏弱区域；
  3) 当前版本更偏向“先给出可运行基线”，后续应结合样本统计分布、研究区经验和实验效果再做校准。
- 后续建议：优先对每个年份的 NDVI、VV、VH 做分布统计或分位数分析，再决定是否把这些固定阈值改成“按数据分位数自动生成”的阈值版本。

### 2026-04-18（gee_downloader 修改前后数据一致性判断）
- 用户询问：此前修改过 `gee_downloader.py`，修改前后下载得到的数据是否会有区别。
- 当前判断：会分两种情况。
  1) 如果比较的是“相同年份、相同月份、相同 ROI、相同数据源与相同筛选条件”下成功导出的月份，则像元数值理论上应基本一致，差别主要体现在任务是否能顺利提交、空月份是否被跳过、监控输出是否更准确；
  2) 如果比较的是修改后新增下载的年份/月份，或者此前某些月份原本因空集合直接失败、现在被跳过或重新组织下载，那么最终拿到的数据集合会有差别，主要是“是否存在该月文件”和“样本覆盖范围”不同。
- 关键原因：这次对 `gee_downloader.py` 的修改核心是增强批量年份支持、增加 S2/S1 空集合检查、改进任务监控和跳过逻辑，并没有主动更换数据源、波段组合、尺度 `scale=100`、ROI 或融合公式，因此对同一成功月份的导出内容影响应很小。
- 风险提醒：如果某个月原脚本虽然提交了任务，但其实因空集合在云端失败，而修改后脚本直接在本地判断为空并跳过，那么“旧版有失败任务记录 / 新版无该文件”会造成数据清单差异，但这不是像元值变化，而是数据可用性变化。

### 2026-04-18（4 月数据是否可并入原有 5–9 月数据集）
- 用户询问：此前用旧版脚本下载了 5–9 月数据，今天用修改后的脚本下载了 4 月数据，是否可以把 4 月数据融入原有数据集合。
- 当前结论：原则上可以并入，但前提是同一年份下 4–9 月数据满足“空间范围一致、波段顺序一致、分辨率一致、特征定义一致”。
- 当前判断依据：虽然下载脚本做过鲁棒性增强，但并未更换 ROI、数据源、波段构成、融合方式和导出尺度，因此若旧版 5–9 月和新版 4 月都成功导出，其数据定义应保持兼容。
- 需要注意的实际问题：
  1) 并入后时间步将从 5 个月变成 6 个月，因此 `data_processor.py`、`main.py`、`config.py` 中涉及输入月份数 / `window_in` 的位置都要同步；
  2) 如果某些年份 4 月成功、5–9 月不完整，或 4 月缺失，就不能直接构成统一长度时序；
  3) 最好抽查 1–2 个年份，确认 4 月 TIFF 与 5–9 月 TIFF 的尺寸、波段数、数值范围一致后再正式并入。
- 当前建议：可以把 4 月数据作为“4–9 月扩展版”单独做一套实验，与原有 `5–9 月` 基线对比，而不是直接覆盖原先主数据集。

### 2026-04-18（进入模型训练阶段的判断与模型定位）
- 用户说明：目前已分别用 `data_processor_y_threshold.py` 和 `data_processor_y.py` 生成两类伪标签，询问是否应该进入模型运行阶段，以及当前模型是否适合“干旱监测 + 未来预测”目标。
- 当前建议：可以进入模型训练/评估阶段，但建议把它明确视为“干旱监测/分类基线模型”，暂时不要把现有结果直接表述为严格意义上的未来预测结果。
- 对当前模型用途的判断：
  1) 现有 `main.py + ConvLSTM / TrajGRU` 链路主要是基于输入时序影像学习像元级干旱等级分类；
  2) 若标签来自最后一个月特征生成的伪标签，则当前训练任务本质上更接近“时序辅助的当期干旱状态识别/监测”；
  3) 它可以作为未来预测研究的基础，但当前配置还不足以严格证明模型已完成真正的未来月份预测。
- 为什么说还不是严格未来预测：
  1) 当前标签构造通常仍依赖输入序列最后一个月的特征；
  2) 训练集/验证集/测试集主要按年份划分，而不是显式设置“输入前几个月，预测后几个月”；
  3) 现有 `window_in=5` 更像用整段时序去判别该段对应的干旱等级，而不是做 lead-time forecasting。
- 当前最合理的下一步：
  1) 先分别用两套伪标签训练和评估，比较 KMeans 标签版与硬阈值标签版谁更稳定；
  2) 把这一阶段结果定位为“干旱监测/识别实验”；
  3) 若后续目标转向真正未来预测，再重构任务为类似“输入 4–7 月，预测 8–9 月”或“输入前 5 个时相，预测下 1 个时相/下 2 个时相”的监督设计。

### 2026-04-18（main.py 支持两套标签训练切换）
- 用户要求：检查 `zyk_drought_monitor` 训练链路代码，并根据两套伪标签训练模型。
- 已完成处理：
  1) 重写 `main.py`，新增 `LABEL_MODE` 开关，可在 `kmeans` 与 `threshold` 两套标签方案之间切换；
  2) 新版训练入口支持按年份自动查找 `dataset_X_<year>.pt` 与对应的标签文件，并兼容多个候选目录；
  3) 模型权重文件名会自动附加标签方案后缀，例如 `drought_trajgru_best_kmeans.pth`、`drought_trajgru_best_threshold.pth`，避免覆盖；
  4) 当前默认训练配置为：训练集 2021–2022、验证集 2023、测试集 2024，模型默认 `traj_gru`。
- 使用建议：
  1) 先设置 `LABEL_MODE = 'kmeans'` 跑一轮；
  2) 再改为 `LABEL_MODE = 'threshold'` 跑一轮；
  3) 后续用评估脚本或结果文件对比两套标签方案的 Accuracy / Macro-F1 / 混淆矩阵。

### 2026-04-18（训练已正常启动与底部输出解释）
- 用户反馈训练界面已出现 epoch 日志，询问“现在是不是可以了”以及最下面一行的含义。
- 当前判断：训练已经正常启动，数据加载、类别权重计算、epoch 迭代都在正常进行中。
- 对底部输出的解释：类似 `val:3/549/109` 这类显示本质上是 notebook 输出被截断/刷新过程中的中间进度文本，不是新的报错。
- 训练循环里真正有意义的进度信息主要是：
  1) `Epoch:x/50`：当前第几轮训练；
  2) `Train_loss / Accuracy / Macro-F1`：训练集指标；
  3) `val:a/b`：验证集当前 batch 进度；
  4) 最终会在 early stop 或最后一轮时输出最佳 epoch 与验证结果。

### 2026-04-18（模型权重保存位置说明）
- 用户在一次成功训练结束后询问：模型权重结果保存在哪里、叫什么名字。
- 当前代码逻辑：训练完成后会执行 `torch.save(model.state_dict(), save_path)`，其中 `save_path = os.path.join(OUTPUT_DIR, save_name)`。
- 当前默认保存目录：`OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/data'`。
- 命名规则：
  1) 若 `ACTIVE_MODEL = 'convlstm'` 且 `LABEL_MODE = 'kmeans'`，则文件名为 `drought_convlstm_best_kmeans.pth`；
  2) 若 `ACTIVE_MODEL = 'convlstm'` 且 `LABEL_MODE = 'threshold'`，则文件名为 `drought_convlstm_best_threshold.pth`；
  3) 若 `ACTIVE_MODEL = 'traj_gru'`，则文件名会对应为 `drought_trajgru_best_<label_mode>.pth`。
- 因此用户这次截图对应的权重文件应保存为：`/root/autodl-tmp/zyk_drought_monitor/data/drought_convlstm_best_kmeans.pth`。

### 2026-04-18（切换 threshold 标签版训练与结果对比说明）
- 用户要求：继续跑 `threshold` 标签版，但只需要指出修改哪里，不需要直接代为操作；同时说明如何对比两套模型结果。
- 操作说明：
  1) 在 `main.py` 顶部把 `LABEL_MODE = 'kmeans'` 改成 `LABEL_MODE = 'threshold'`；
  2) 保持 `ACTIVE_MODEL` 不变（若当前想做同模型下标签对比，就继续用同一个模型，例如 `convlstm`）；
  3) 重新运行训练入口后，模型会自动保存为带 `threshold` 后缀的权重文件，例如 `drought_convlstm_best_threshold.pth`。
- 对比建议：
  1) 固定模型结构不变，先比较 `kmeans` 与 `threshold` 两套标签在验证集/测试集上的 `Accuracy` 与 `Macro-F1`；
  2) 重点看 `Macro-F1` 与混淆矩阵，因为这能更好反映中旱/重旱等少数类是否被识别；
  3) 若两套标签结果接近，则优先选择语义更稳定、类别分布更合理的一套；若差异明显，则后续可再比较不同模型（ConvLSTM vs TrajGRU）。

### 2026-04-18（ConvLSTM 下两套标签方案的首次对比结论）
- 用户已跑出 `ConvLSTM + threshold` 结果，并希望与此前的 `ConvLSTM + kmeans` 进行对比。
- 当前已知结果：
  1) `ConvLSTM + kmeans`
     - 最佳验证：Acc ≈ 0.6381，Macro-F1 ≈ 0.6143
     - 测试：Acc ≈ 0.4782，Macro-F1 ≈ 0.3150
  2) `ConvLSTM + threshold`
     - 最佳验证：Acc ≈ 0.9497，Macro-F1 ≈ 0.8170
     - 测试：Acc ≈ 0.9044，Macro-F1 ≈ 0.7027
- 对比结论：在当前 ConvLSTM 与当前数据划分下，`threshold` 标签方案明显优于 `kmeans` 标签方案，且优势非常大，不仅 Accuracy 更高，Macro-F1 也显著提升。
- 含义解释：
  1) 当前这批数据上，硬阈值构造出来的监督信号比 KMeans 聚类标签更稳定；
  2) `threshold` 方案不仅更容易学，而且泛化到测试集也更好；
  3) 因为测试集 Macro-F1 从约 0.315 提升到约 0.703，这已经不只是小波动，而是监督标签质量差异在模型上被明显放大了。
- 风险提醒：
  1) `threshold` 标签的类别权重中第 4 类权重非常高（约 73.23），说明某个类别仍然非常稀少，后续应继续查看混淆矩阵与类别分布；
  2) 当前优劣结论仅代表“相对于这两套伪标签，threshold 更适合作为当前监测实验标签”，不等于标签本身就是真实干旱真值。
- 当前建议：后续应优先以 `threshold` 标签方案作为主线，接下来再比较 `TrajGRU + threshold` 与 `ConvLSTM + threshold`，判断是“标签方案”还是“模型结构”更影响最终表现。

### 2026-04-18（从旱情监测扩展到旱情预测的路线建议）
- 用户说明：目标不仅是旱情监测，还希望同时进行旱情预测。
- 当前判断：现有链路已经比较适合做“监测/识别”，若要进一步做“预测”，关键不只是换模型，而是要重构监督任务。
- 核心思路：把当前“输入整段时序 → 识别当前/末月旱情”的任务，改造成“输入前几个月 → 预测未来月份旱情”的 lead-time 预测任务。
- 推荐路线：
  1) 短期：保留当前 `threshold` 标签作为监测主线，继续完成模型对比与稳定评估；
  2) 中期：设计真正的预测样本，例如“输入 4–7 月，预测 8 月”或“输入 4–8 月，预测 9 月”；
  3) 长期：若想同时输出监测与预测，可考虑共享编码器 + 双任务头，或先训练监测模型，再单独构建预测模型作为第二阶段。
- 当前建议：不要把现有监测模型直接当作预测模型使用；更合理的做法是先把“监测”和“预测”拆成两个明确任务，再决定是否做联合建模。

### 2026-04-18（监测模型对比范围与 main.py 扩展）
- 用户询问：`models/baseline/convlstm.py` 是带 attention 的 ConvLSTM，baseline 文件夹中还有其他模型，是否要一起运行对比；若需要，请修改 `main.py`。
- 当前判断：建议纳入统一监测对比的主力模型优先选择：`convlstm_attn`、`convlstm_no_attn`、`traj_gru`。
- 原因：
  1) `convlstm_attn` 与 `convlstm_no_attn` 可以直接回答 attention 机制是否有效；
  2) `traj_gru` 已经适配当前分类任务，且适合做时空建模对比；
  3) `lstm.py`、`u_net.py`、`moving_avg.py` 当前接口或任务形式与现有像元级多分类训练链路不完全一致，暂不建议直接纳入同一轮无改动对比。
- 已完成代码修改：
  1) `main.py` 现支持 `ACTIVE_MODEL = 'convlstm_attn' | 'convlstm_no_attn' | 'traj_gru'`；
  2) 当选择 `convlstm_no_attn` 时，会复用同一套 ConvLSTM 结构但关闭输入 attention；
  3) 模型权重文件名会自动区分，例如 `drought_convlstm_attn_best_threshold.pth`、`drought_convlstm_no_attn_best_threshold.pth`、`drought_traj_gru_best_threshold.pth`。
- 当前建议：先固定 `LABEL_MODE = 'threshold'`，然后依次比较三组结果：
  1) `convlstm_attn + threshold`
  2) `convlstm_no_attn + threshold`
  3) `traj_gru + threshold`

### 2026-04-18（监测/预测任务可扩展模型建议）
- 用户询问：除了当前项目中已有的模型外，还有哪些模型适合当前农业旱情监测与未来预测任务。
- 当前建议：可从以下几类模型扩展，而不是盲目全部尝试。
- 适合优先考虑的监测/预测模型方向：
  1) `PredRNN / PredRNN++`：适合更强的时空序列建模，尤其是未来帧/未来状态预测；
  2) `ConvGRU`：比 ConvLSTM 更轻量，可作为时空递归模型对照组；
  3) `UNet + ConvLSTM` 或 `CNN + LSTM/GRU`：先提空间特征，再做时序建模，适合把监测与预测分层处理；
  4) `Temporal Convolutional Network (TCN)` / 3D-CNN：适合做固定窗口时空分类或短时预测；
  5) `Transformer / TimeSformer / Swin + Temporal Fusion`：在样本量足够时可建模更长时序依赖，但训练成本更高；
  6) `Graph Neural Network (GNN)` 结合时序模块：若后续想显式建模空间邻接关系，可作为高级扩展方向。
- 当前阶段最务实的建议：
  1) 先把现有 `convlstm_attn`、`convlstm_no_attn`、`traj_gru` 比完；
  2) 若要加新模型，优先考虑实现难度和你当前任务最匹配的 `ConvGRU` 或 `PredRNN`；
  3) Transformer 类模型可作为后续提升方向，但不建议在当前样本规模和标签体系还在稳定阶段时过早引入。
- 任务匹配提醒：
  1) 若重点是“监测/识别”，优先考虑分类分割型时空模型；
  2) 若重点转向“未来预测”，更适合引入显式序列预测模型，如 `PredRNN`、`ConvGRU`、编码器-解码器时空网络。

### 2026-04-18（已新增 ConvGRU baseline）
- 用户要求：增加 `ConvGRU` 模型，并放入 `models/baseline` 文件夹中。
- 已完成内容：
  1) 新增文件 `models/baseline/convgru.py`，实现了适配当前像元级多分类监测任务的 `ConvGRU`；
  2) 新增 `configs/config.py` 中的 `convgru` 配置，参数设置与当前 ConvLSTM 基本对齐；
  3) 更新 `main.py`，现在 `ACTIVE_MODEL` 支持 `convgru`，可直接切换训练；
  4) 权重文件命名规则已接入，后续会保存为 `drought_convgru_best_<label_mode>.pth`。
- 当前建议：先在 `LABEL_MODE='threshold'` 下比较 `convlstm_attn`、`convlstm_no_attn`、`convgru`、`traj_gru` 四组结果，再决定下一步是否扩展到更复杂的预测模型。

### 2026-04-18（已新增独立预测训练入口 forecast_main.py）
- 用户要求：在不修改现有 `main.py` 的前提下，新建一个文件，使模型具备“干旱预测”训练能力。
- 已完成内容：
  1) 新建文件 `forecast_main.py`，作为独立于监测脚本的预测版训练入口；
  2) 当前预测任务定义为：输入前 4 个月特征，预测第 5 个月的旱情等级；
  3) 支持的模型包括：`convlstm_attn`、`convlstm_no_attn`、`convgru`、`traj_gru`；
  4) 保存权重文件名会带 `forecast` 前缀，例如 `drought_forecast_convgru_best_threshold.pth`。
- 当前实现说明：
  1) 预测脚本直接复用现有按年份保存的 `dataset_X_YYYY.pt` 与 `dataset_Y_YYYY(_threshold).pt`；
  2) 样本构造方式改为仅截取前 4 个时间步作为输入，以此区别于原监测任务；
  3) 标签仍使用当前年份窗口的末月标签，因此该版本可作为“先行预测版基线”。
- 风险提醒：该脚本已经把任务形式改成“前几个月 → 末月旱情”，但若后续想做更严格的跨月 lead-time 预测，最好进一步把标签生成逻辑显式改成与未来月份一一对齐。

### 2026-04-18（已新增监测任务统一评估与空间可视化脚本）
- 用户要求：回到 `main.py` 的监测任务，希望证明不同模型效果，并进行可视化；随后要求“都做吧，给我代码”。
- 已完成内容：
  1) 新建 `monitor_compare.py`，用于统一评估多个监测模型 checkpoint，并自动生成：
     - 文本汇总 `model_summary.txt`
     - 指标柱状图 `metrics_comparison.png`
     - 标准化混淆矩阵图 `confusion_matrices_normalized.png`
  2) 新建 `monitor_spatial_compare.py`，用于在同一个测试样本上可视化：
     - 参考底图（默认最后一个月 NDVI）
     - 真实旱情图
     - 多个模型的预测旱情图
- 当前意义：这样就能同时从“定量指标”和“空间分布图”两条线证明不同模型在监测任务上的优劣。

### 2026-04-18（监测可视化样本的选择、获取与调试方法）
- 用户希望：把“用于空间可视化的样本如何选择、如何获取、如何调试”的详细方法记录到 `conversation_experiment_log.md` 中。
- 当前脚本依据：`monitor_spatial_compare.py` 中的样本由 `--sample_index` 指定，本质上是从 `dataset_X_<year>.pt` 与 `dataset_Y_<year>*.pt` 中按索引直接取第 `i` 个样本。
- 关键代码逻辑：
  1) 先加载 `x_all = torch.load(x_path)` 与 `y_all = torch.load(y_path)`；
  2) 检查 `sample_index` 是否在 `[0, x_all.shape[0)-1]` 范围内；
  3) 通过 `x_sample = x_all[sample_index]`、`y_true = y_all[sample_index]` 取得目标样本；
  4) 默认参考底图取 `x_sample[-1, 0]`，即最后一个月的 NDVI 通道。
- 如何获取可选样本范围：
  1) 直接看测试集形状，例如 `X=torch.Size([40, 5, 4, 128, 128])` 表示共有 40 个样本；
  2) 此时 `sample_index` 的合法范围就是 `0~39`。
- 推荐的样本选择策略：
  1) 不要优先挑全是“无旱”的样本，因为不同模型画出来差异通常不明显；
  2) 优先挑标签中同时包含 `0/1/2/3` 多个类别的样本；
  3) 若想突出模型差异，可优先挑不同模型预测结果差异较大的样本。
- 快速调试方法：
  1) 先手动尝试几个索引，例如 `0、5、10、15、20`；
  2) 观察哪一个样本的真实旱情图类别更丰富、空间斑块更明显；
  3) 再固定该样本用于论文图或结果展示。
- 更稳妥的调试方法：先统计每个样本包含哪些类别，例如执行：
  `y = torch.load('/root/autodl-tmp/data_proc/dataset_Y_2025_threshold.pt')`
  然后遍历 `torch.unique(y[i])`，筛出包含多个类别的样本索引，再传给 `--sample_index`。
- 参考底图的调试方法：
  1) 默认 `feature_time_index=-1`、`feature_channel_index=0`，表示最后一个月 NDVI；
  2) 若想查看 VV 或 VH，可分别设为 `feature_channel_index=1` 或 `2`；
  3) 若想查看更早月份，可把 `feature_time_index` 改成 `0/1/2/...`。
- 当前建议：先筛出 1–3 个类别丰富的测试样本，再用 `monitor_spatial_compare.py` 生成多模型空间预测图，最终挑选最能体现模型差异的一张作为核心可视化结果。

### 2026-04-18（monitor_spatial_compare.py 中文字体修复）
- 用户反馈：空间预测对比图中的中文显示为方格，希望使用 `SimHei.ttf` 实现中文显示。
- 已完成内容：
  1) 在 `monitor_spatial_compare.py` 中加入 `font_manager`；
  2) 增加 `setup_chinese_font()`，默认加载 `/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf`；
  3) 在 `main()` 开始处自动启用中文字体设置，使标题、类别标签、色标标签都能正常显示中文。

### 2026-04-18（关于旱情预测应如何开展的详细路线）
- 用户希望：系统梳理“关于旱情预测，我应该怎么做”，并把详细说明写入 `conversation_experiment_log.md`。
- 当前总体判断：用户当前已经形成了较完整的“旱情监测”链路，但“旱情预测”不能直接等同于当前监测模型，需要从任务定义、样本构造、标签组织、模型选择、评估方式五个层面重新设计。

#### 一、先明确“监测”和“预测”的区别
- 当前监测任务更接近：
  1) 输入 5 个月时序特征；
  2) 输出当前窗口末月或当前时段的旱情等级图；
  3) 本质是时序辅助下的空间分类/识别。
- 真正的旱情预测任务应改成：
  1) 输入前几个月特征；
  2) 输出未来一个月或未来几个月的旱情等级图；
  3) 核心是 lead-time forecasting（超前预测）。
- 因此当前最关键的变化不是先换模型，而是先改监督任务定义。

#### 二、推荐的预测任务定义方式
- 推荐先从最简单、最稳妥的单步预测开始：
  1) 输入 4–7 月，预测 8 月旱情；
  2) 或输入 5–8 月，预测 9 月旱情。
- 若当前每个年度样本已经整理成固定 5 个时相窗口，则可以先做：
  1) 输入前 4 个时间步；
  2) 预测第 5 个时间步对应的旱情等级。
- 这样做的好处是：
  1) 能最大程度复用现有数据张量结构；
  2) 任务定义清晰；
  3) 便于后续扩展到“预测未来 2 个月”“滚动预测”等更复杂方案。

#### 三、预测任务的数据组织应该怎么改
- 当前监测版数据通常可表示为：
  1) `X: (N, 5, 4, 128, 128)`，表示 5 个时间步、4 个通道；
  2) `Y: (N, 128, 128)`，表示末月干旱等级图。
- 若改成预测，应显式保证：
  1) 输入不能包含目标月份本身；
  2) 标签必须来自未来月份；
  3) 训练、验证、测试仍按年份拆分，避免时间泄漏。
- 推荐的数据构造方式：
  1) 若原始 5 时相为 `t1, t2, t3, t4, t5`；
  2) 则预测样本构造为 `X_forecast = [t1, t2, t3, t4]`；
  3) `Y_forecast = drought_label(t5)`。
- 更严格的版本：后续应让 `data_processor_y_threshold.py` 与 `data_processor_y.py` 支持“指定目标月份生成标签”，而不是默认始终基于最后一个月生成标签。

#### 四、标签层面应怎么处理
- 当前建议继续以 `threshold` 作为预测任务的主线标签方案，原因是：
  1) 在监测任务中，`threshold` 已明显优于 `kmeans`；
  2) `threshold` 的语义更稳定，更适合作为未来预测目标；
  3) 聚类标签更容易引入随年份波动的伪标签噪声。
- 但需要注意：
  1) 预测任务中的标签必须严格对应未来月份；
  2) 不能只是“继续沿用当前窗口标签”而默认认为就是预测；
  3) 若后续要做更标准的 forecasting，必须把标签生成逻辑显式绑定到目标月份。

#### 五、模型层面应该怎么推进
- 当前最务实的推进顺序建议为：
  1) 先固定标签方案为 `threshold`；
  2) 先在监测任务里选出最稳的模型主线；
  3) 再将该模型迁移到预测任务；
  4) 之后再比较不同模型在预测任务上的差异。
- 当前推荐的预测模型优先级：
  1) `ConvGRU`：结构轻量，易于迁移到“前几个月 → 未来月份”的任务；
  2) `ConvLSTM`：适合作为最稳定、最容易解释的预测基线；
  3) `TrajGRU`：可作为更强时空动态建模对照组；
  4) 后续若要进一步增强，可考虑 `PredRNN / PredRNN++`。
- 当前不建议一开始就上 Transformer 类模型，原因是：
  1) 样本规模尚不算特别大；
  2) 标签体系仍在稳定阶段；
  3) 先做出清晰、可信的 forecasting baseline 更重要。

#### 六、评估预测任务时不能只看 Accuracy
- 预测任务建议继续保留当前分类指标：
  1) Accuracy；
  2) Macro-F1；
  3) 混淆矩阵。
- 但相比监测任务，更建议额外关注：
  1) 不同 lead time 下的性能变化；
  2) 对中旱/重旱类别的召回情况；
  3) 空间分布预测是否存在明显滞后或过平滑。
- 若后续扩展到多步预测，还建议记录：
  1) `t+1` 月性能；
  2) `t+2` 月性能；
  3) 随预测步长增长，性能下降的趋势。

#### 七、推荐的实验推进顺序
- 第一阶段：监测主线稳定化
  1) 在 `threshold` 标签下完成 `convlstm_attn`、`convlstm_no_attn`、`convgru`、`traj_gru` 的统一对比；
  2) 通过指标、混淆矩阵、空间预测图选出监测任务主模型。
- 第二阶段：建立预测基线
  1) 先用当前独立脚本 `forecast_main.py` 做“前 4 个时间步 → 第 5 个时间步旱情”的基线预测；
  2) 固定标签方案为 `threshold`；
  3) 优先比较 `ConvGRU` 与 `ConvLSTM` 两种预测版结果。
- 第三阶段：做更严格的 forecasting 数据重构
  1) 让标签生成脚本支持“指定目标月份生成标签”；
  2) 用更明确的月份对应关系构建样本，例如输入 4–7 月、预测 8 月；
  3) 若数据允许，再扩展到多步滚动预测。

#### 八、当前最推荐的实际落地方案
- 若用户现在就想启动旱情预测研究，建议按以下顺序实际执行：
  1) 继续保留当前 `main.py` 作为监测主线；
  2) 使用 `forecast_main.py` 作为独立的预测入口，不与监测任务混用；
  3) 先跑 `convgru + threshold` 与 `convlstm_no_attn + threshold` 两个预测基线；
  4) 若预测结果已经具有可解释性，再继续优化 attention 或引入更复杂模型；
  5) 若预测结果不理想，优先先检查数据与标签定义，而不是先堆更复杂网络。

#### 九、当前应该避免的误区
- 不建议把“监测做得好”直接等同于“预测也会好”；
- 不建议直接把当前末月分类任务硬说成未来预测；
- 不建议在标签定义还不严格的情况下过早上复杂模型；
- 不建议同时大改标签、模型、数据切分，否则很难知道到底是哪一部分起作用。

#### 十、当前阶段的总建议
- 当前最合理的理解方式是：
  1) 监测任务负责证明现有特征和标签体系是可学的；
  2) 预测任务负责证明这些时序特征是否对未来旱情具有前瞻性；
  3) 两者应分开建模、分开评估，但可共享同一套基础特征与标签体系。
- 当前建议：以 `threshold` 为主线标签，监测先完成四模型对比，预测先用 `forecast_main.py` 建立单步预测基线，后续再逐步升级到更标准的 lead-time forecasting。

### 2026-04-18（forecast_main.py 预测效果验证与可视化）
- 用户希望：验证 `forecast_main.py` 的预测效果，并完成预测任务可视化，同时要求引入中文字体。
- 已完成内容：新增两套预测任务专用脚本，均默认加载 `/root/autodl-tmp/zyk_drought_monitor/SimHei.ttf` 以保证中文正常显示。

#### 1. 预测任务统一评估脚本
- 新增文件：`forecast_compare.py`
- 主要功能：
  1) 读取测试年份的 `.pt` 数据；
  2) 按 `forecast_main.py` 当前定义自动构造成预测样本（前 4 个时间步输入、预测第 5 个时间步标签）；
  3) 批量加载多个预测模型权重；
  4) 输出 Loss、Accuracy、Macro-F1、Weighted-F1；
  5) 生成中文指标柱状图；
  6) 生成中文混淆矩阵图；
  7) 保存文本版分类报告汇总。
- 生成结果目录默认位于：`results/forecast_compare/`
- 主要输出文件包括：
  1) `forecast_model_summary.txt`
  2) `forecast_metrics_comparison.png`
  3) `forecast_confusion_matrices_normalized.png`

#### 2. 预测任务空间对比可视化脚本
- 新增文件：`forecast_spatial_compare.py`
- 主要功能：
  1) 指定测试样本 `sample_index`；
  2) 使用预测任务输入窗口生成各模型预测图；
  3) 同时显示参考底图、真实未来旱情图、各模型预测图；
  4) 使用中文标题、中文类别标签与中文色标说明。
- 默认输出文件：`results/forecast_compare/forecast_spatial_prediction_compare.png`

#### 3. 当前建议的验证方式
- 建议按以下顺序验证 `forecast_main.py` 的预测效果：
  1) 先分别训练得到 `forecast` 前缀的模型权重；
  2) 用 `forecast_compare.py` 比较整体指标；
  3) 再用 `forecast_spatial_compare.py` 挑选典型样本做空间对比；
  4) 最终从“整体指标 + 混淆矩阵 + 空间图”三个层面综合判断模型预测能力。
- 当前这一套流程已经和监测任务的验证方式对应起来，便于后续统一汇报。

### 2026-04-18（forecast_main.py 从数据构造层面如何提升预测效果）
- 用户希望：进一步分析 `forecast_main.py` 还能从数据构造上怎么改，以提升预测效果；同时考虑是否需要重新获取数据、还需要补充哪些数据，并把完整分析写入日志，便于次日继续实施。
- 当前总体判断：现有 `forecast_main.py` 已经建立了一个“前 4 个时间步 → 第 5 个时间步旱情”的单步预测基线，但它本质上仍是一个简化版 forecasting 任务。当前预测效果不理想，更可能首先是数据构造与信息量不足的问题，而不仅仅是模型结构的问题。

#### 一、当前 `forecast_main.py` 的数据构造方式存在哪些局限
- 当前做法本质上是：
  1) 从 `dataset_X_year.pt` 中取前 4 个时间步作为输入；
  2) 用同一窗口下的 `dataset_Y_year*.pt` 作为目标；
  3) 默认认为该标签代表第 5 个时间步未来旱情。
- 这种构造适合做“最初版预测基线”，但有几个明显局限：
  1) 输入窗口长度较短，历史记忆不足；
  2) 标签生成逻辑没有显式绑定到“未来月份”；
  3) 样本之间仍是固定窗口，缺少更密集的滑动构样；
  4) 特征仍以当前遥感监测变量为主，对未来旱情的驱动信息不足；
  5) 训练样本量可能偏少，难以支撑更稳的 forecasting 学习。

#### 二、优先建议改进的不是模型，而是样本构造方式
- 当前最应该优先调整的是：
  1) 让输入窗口和目标月份的关系更严格；
  2) 让每个样本明确对应“预测 lead time”；
  3) 让样本数量通过滑动窗口显著增加；
  4) 让输入包含更多与未来旱情相关的驱动信息。
- 换句话说，下一阶段最核心的工作是“重构 forecasting 数据集”，而不是继续只在当前 `.pt` 上反复换模型。

#### 三、从时间组织上，数据应怎么重新构造
- 当前推荐的标准化组织方式是：按自然月份重新整理为月序列，而不是只保留一个固定 5 时相切片。
- 推荐做法：
  1) 对每个空间样本（tile）保留连续月份序列；
  2) 每个月都有对应的特征张量 `X_t`；
  3) 每个月都有对应的旱情标签 `Y_t`；
  4) 再通过滑动窗口构造 forecasting 样本。
- 例如：若已有 2021–2025 年连续月度数据，则可构造成：
  1) 输入 `X[t-3], X[t-2], X[t-1], X[t]`；
  2) 预测 `Y[t+1]`；
  3) 这就是严格意义上的 `t+1` 单步预测。
- 后续还可以继续扩展：
  1) 输入过去 6 个月，预测未来 1 个月；
  2) 输入过去 6 个月，预测未来 2 个月；
  3) 比较不同输入长度、不同 lead time 对结果的影响。

#### 四、建议把“固定样本”改成“滑动窗口样本”
- 当前固定 5 时相窗口的样本量通常有限，而 forecasting 特别依赖样本量。
- 推荐改法：
  1) 若有连续 24 个月数据，就不要只构造一个样本；
  2) 而应通过滑动窗口生成多个样本；
  3) 比如窗口长度为 4，预测步长为 1，则一个长度为 24 的时间序列可生成约 19 个样本。
- 好处：
  1) 样本量明显增加；
  2) 模型能看到更多季节变化与状态转移；
  3) 有助于提升 forecasting 学习稳定性。

#### 五、输入长度建议不要只固定为 4 个月
- 当前 `FORECAST_INPUT_STEPS = 4` 只是一个可行起点，但不一定最优。
- 建议后续做输入长度对比实验：
  1) 3 个月输入；
  2) 4 个月输入；
  3) 6 个月输入；
  4) 9 个月输入。
- 原因：
  1) 干旱具有累积和滞后效应；
  2) 植被、土壤湿度、降水异常对未来旱情的影响可能跨越多个时间尺度；
  3) 过短的输入窗口可能只能学到短期波动，学不到季节性和累积效应。
- 当前建议：下一版 forecasting 数据集至少做出 `4个月输入` 与 `6个月输入` 两套版本。

#### 六、标签生成逻辑要显式绑定目标月份
- 当前最重要的改动之一：标签脚本需要支持“指定月份生成标签”。
- 建议后续把标签生成流程改成：
  1) 给定一个目标月份 `target_month`；
  2) 从该月对应的原始数据计算干旱等级图；
  3) 生成 `Y_target_month`；
  4) 再与过去若干个月输入配对。
- 这样才能确保：
  1) 输入确实只来自过去；
  2) 标签确实只来自未来；
  3) forecasting 任务定义严格成立。
- 当前阶段仍建议以 `threshold` 作为主线标签方案，不建议优先用 `kmeans` 做 forecasting 主实验。

#### 七、从空间样本组织上也可以继续优化
- 当前若一个 tile 内部差异极大，模型容易输出过平滑结果。
- 可考虑的空间层面改进：
  1) 检查 tile 是否过大，导致内部异质性太强；
  2) 适当增加样本数量，而不是只依赖少量固定 tile；
  3) 对类别极度单一的 tile 降低采样比例；
  4) 优先保留类别更丰富、变化更明显的区域用于训练。
- 目标不是人为挑“好看的样本”，而是让训练集更充分覆盖多样化干旱状态转移。

#### 八、仅靠现有 4 个输入通道，对未来预测可能不够
- 当前监测特征主要是：
  1) NDVI；
  2) VV；
  3) VH；
  4) VVVH。
- 这些特征对“当前旱情识别”是有帮助的，但对“未来旱情预测”的前瞻性可能有限。
- forecasting 更需要引入对未来具有驱动意义的变量，而不仅是当前遥感响应变量。

#### 九、下一步最值得重新获取或补充的数据
- 若目标是把 forecasting 做得更扎实，建议优先补充以下数据：

##### 1. 月尺度降水数据
- 重要性：最高。
- 原因：降水是未来旱情变化的直接驱动因子之一。
- 推荐用途：
  1) 当前月降水；
  2) 前 1–3 个月累计降水；
  3) 降水距平或降水异常值。
- 如果只能优先加一类新数据，建议先加降水。

##### 2. 地表温度或气温数据
- 重要性：很高。
- 原因：温度影响蒸散发强度，与干旱发展关系密切。
- 推荐用途：
  1) 月均温；
  2) 月最高温；
  3) 温度距平。

##### 3. 土壤湿度数据
- 重要性：很高。
- 原因：土壤湿度对旱情具有明显的滞后与记忆效应，是 forecasting 的关键变量之一。
- 推荐用途：
  1) 表层土壤湿度；
  2) 根区土壤湿度；
  3) 过去数月土壤湿度变化趋势。

##### 4. 蒸散发 / 潜在蒸散发数据
- 重要性：高。
- 原因：干旱演变不仅与水分输入有关，也与水分消耗有关。
- 推荐用途：
  1) 月蒸散发；
  2) 潜在蒸散发；
  3) 干旱水分亏缺相关组合指标。

##### 5. 更标准的干旱指数数据
- 推荐考虑：
  1) SPI；
  2) SPEI；
  3) VCI / TCI / VHI；
  4) 土壤湿度异常指数。
- 用途：
  1) 作为附加输入特征；
  2) 或作为辅助标签质量检验依据；
  3) 也可帮助解释模型预测是否符合旱情机理。

##### 6. 更长时间范围的历史数据
- 当前若只覆盖较短年份，forecasting 很难学到稳定的年际变化。
- 建议尽量扩展到更长历史期，例如：
  1) 至少 5 年以上连续月序列；
  2) 若条件允许，扩展到 8–10 年更理想。
- 长时间序列的价值在于：
  1) 样本量增加；
  2) 极端年份增多；
  3) 模型更容易学到跨年变化与季节规律。

#### 十、如果明天要真正开始做，推荐的实施顺序
- 建议按下面顺序推进，而不是一次性全改：

##### 第一步：梳理现有原始时间序列资源
- 明天首先确认：
  1) 当前原始 GEE/遥感数据是否按月可回溯；
  2) 每个月是否都能重新导出 NDVI、VV、VH、VVVH；
  3) 标签是否能按指定月份重新生成；
  4) 是否已经具备降水、温度、土壤湿度等外部数据源。

##### 第二步：先重做一个更严格的 forecasting 数据集 V2
- 建议先不要一次加入所有新变量，而是先做“结构更合理”的 V2 数据集：
  1) 连续月序列；
  2) 滑动窗口构样；
  3) 输入过去 4 或 6 个月；
  4) 标签明确绑定未来 1 个月；
  5) 仍只用当前 4 个遥感通道 + `threshold` 标签。
- 这样可以先验证：仅靠样本构造优化，预测是否已经改善。

##### 第三步：再做带气象驱动的 forecasting 数据集 V3
- 在 V2 基础上继续加入：
  1) 降水；
  2) 温度；
  3) 土壤湿度；
  4) 蒸散发。
- V3 的目标是验证：驱动因子是否显著提升未来旱情预测能力。

##### 第四步：最后再做模型升级
- 当 V2、V3 数据集都建立后，再决定是否继续：
  1) 优化 attention；
  2) 尝试更强的时空预测模型；
  3) 做多步预测。
- 这样实验链路更清晰，后续写分析时也更容易说明“性能提升是来自数据构造还是来自模型升级”。

#### 十一、当前最建议你明天优先完成的具体事项
- 若明天时间有限，优先完成以下 5 件事：
  1) 确认现有原始数据是否支持按月重建输入与标签；
  2) 列出当前每个年份实际可用的连续月份范围；
  3) 评估能否额外获取降水、温度、土壤湿度数据；
  4) 设计 forecasting 数据集 V2 的样本结构（输入月数、预测步长、滑动窗口规则）；
  5) 明确新的输出文件命名规则，例如 `forecast_v2_X_*.pt`、`forecast_v2_Y_*.pt`。

#### 十二、当前阶段的总建议
- 当前不建议再仅围绕现有 `forecast_main.py` 调模型超参数，而应优先做：
  1) 更严格的时间对齐；
  2) 更丰富的滑动窗口样本；
  3) 更长输入记忆；
  4) 更强的外部驱动变量补充。
- 当前最值得优先补采的数据顺序建议为：
  1) 降水；
  2) 温度；
  3) 土壤湿度；
  4) 蒸散发；
  5) 更标准的干旱指数。
- 若只做一件最关键的事，建议优先把 forecasting 数据集从“固定 5 时相切片”升级为“连续月序列 + 滑动窗口 + 明确未来目标月份”的标准化数据组织方式。

### 2026-04-19（数据获取与预处理：forecasting V2 落地方案）
- 用户补充了更明确的约束与目标：
  1) 当前只能稳定使用 Sentinel-1 和 Sentinel-2 两类遥感数据；
  2) 可以考虑基于这两类数据增加更多指数，而不只使用 NDVI；
  3) 若要扩展月份，需要把 2021–2025 年更多月份重新下载；
  4) 标签脚本需要支持“指定月份生成标签”；
  5) 希望明确 forecasting 数据集 V2 除上述三点之外是否还有其他必要环节，并直接进行可执行修改；
  6) 要求把这次讨论与操作详细保存在会话日志中。

#### 一、关于目前只能使用 Sentinel-1 / Sentinel-2 的判断
- 当前在数据源受限的情况下，仍然可以推进 forecasting V2，不必等待气象或再分析数据。
- 关键思路是：
  1) 继续以 Sentinel-2 提供植被/水分相关光学指数；
  2) 继续以 Sentinel-1 提供结构/散射相关雷达特征；
  3) 通过增加更多遥感指数来提升输入信息量；
  4) 先把“时间组织方式”做正确，再观察 forecasting 是否改善。
- 结论：即使目前只有 S1 + S2，也完全值得先做 forecasting V2。

#### 二、关于是否需要把 2021–2025 年更多月份重新下载
- 结论：需要。
- 原因：
  1) 当前 forecasting 基线仍主要依赖固定 5 时相切片；
  2) 若要做连续月序列 + 滑动窗口，就必须补齐更长时间范围内的逐月影像；
  3) 否则无法构造严格的 `t-3,t-2,t-1,t -> t+1` 或更长记忆样本。
- 当前建议：
  1) 先覆盖 2021–2025 年每年的 4–9 月；
  2) 若后续云量与数据质量允许，再扩展到 3–10 月；
  3) 这样既能覆盖主要生长季，也方便对干旱演变做连续建模。

#### 三、已对 `gee_downloader.py` 做的实际修改
- 已将 `gee_downloader.py` 改造成更适合 forecasting V2 的批量月尺度下载脚本。
- 当前脚本支持：
  1) 自定义年份列表（默认 2021–2025）；
  2) 自定义月份列表（默认 4–9 月）；
  3) 自定义导出到 Drive 的文件夹；
  4) 自定义导出特征列表；
  5) 自动监控任务状态。
- 在只使用 Sentinel-1 / Sentinel-2 的前提下，当前已支持导出的特征包括：
  1) `NDVI`
  2) `EVI`
  3) `NDMI`
  4) `NDWI`
  5) `MSAVI`
  6) `VV`
  7) `VH`
  8) `VVVH`
  9) `VVDIFFVH`
  10) `RVI`
- 这些特征全部来自 S1 / S2，本次未引入其他外部数据源，符合用户当前约束。

#### 四、关于“标签脚本支持指定月份生成标签”的实际落地
- 已新增脚本：`generate_monthly_threshold_labels.py`
- 该脚本功能为：
  1) 读取一个时间序列 `X_tensor`；
  2) 指定 `target_month_index`；
  3) 仅用该月份对应的 NDVI / VV / VH 生成阈值标签；
  4) 输出新的 `pt` 标签文件。
- 这意味着：
  1) 标签不再被固定绑死在“最后一个月”；
  2) 后续可以为任意目标月份单独生成标签；
  3) 为 forecasting 数据集严格绑定未来月份提供了必要基础。

#### 五、关于 forecasting 数据集 V2 是否只包含那三点
- 结论：不止三点。
- 用户提出的三点是 forecasting V2 的核心基础，但如果想真正形成可训练的 V2 数据集，还至少需要再补一个环节：
  1) 连续月序列构造成张量；
  2) 按指定月份生成标签；
  3) 通过滑动窗口把月序列切成 forecasting 样本；
  4) 最终输出新的 `forecast_v2_X.pt` 与 `forecast_v2_Y.pt`。
- 也就是说，除了“下载更多月份”和“按指定月份生成标签”，还必须有一个“V2 样本构造脚本”。

#### 六、已新增 forecasting V2 样本构造脚本
- 已新增脚本：`build_forecast_v2_dataset.py`
- 该脚本假设：
  1) `X_tensor` 已经是连续月份组织的 `(Batch, Time, Channels, H, W)`；
  2) `Y_tensor` 已经是按时间展开的 `(Batch, Time, H, W)`；
  3) 然后通过 `input_steps + lead_steps` 的滑动窗口生成 forecasting V2 样本。
- 当前输出结果为：
  1) `forecast_v2_X`: `(NewBatch, InputSteps, Channels, H, W)`
  2) `forecast_v2_Y`: `(NewBatch, H, W)`
- 这一步是 forecasting V2 真正区别于当前 `forecast_main.py` 的核心操作。

#### 七、因此，forecasting V2 当前完整链路应理解为 4 个步骤
- 不再是 3 点，而应是下面 4 个步骤：
  1) 用修改后的 `gee_downloader.py` 批量下载 2021–2025 年更多月份的 S1/S2 月尺度特征；
  2) 用影像处理脚本把连续月份 TIFF 堆叠成新的时间序列 `X_tensor`；
  3) 用 `generate_monthly_threshold_labels.py` 按指定月份生成新的时间序列标签；
  4) 用 `build_forecast_v2_dataset.py` 把连续月序列切成 forecasting V2 的滑动窗口样本。
- 这 4 步合起来，才是 forecasting 数据集 V2 的完整落地流程。

#### 八、当前还需要用户明天继续完成的关键工作
- 尽管本次已经把关键脚本补上，但 forecasting V2 真正落地仍需要继续完成以下工作：
  1) 确定 2021–2025 年要下载的具体月份范围；
  2) 实际执行 `gee_downloader.py` 导出任务；
  3) 把导出的 TIFF 整理到统一目录；
  4) 编写或调整连续月份 `X_tensor` 构造逻辑，使其不再只固定 5 时相；
  5) 生成按时间展开的 `Y_tensor` 序列；
  6) 最后再运行 `build_forecast_v2_dataset.py` 输出训练用 pt 文件。

#### 九、关于指数选择的当前建议
- 在当前只用 S1/S2 的前提下，建议先用下面这组特征作为 forecasting V2 输入候选：
  1) `NDVI`
  2) `EVI`
  3) `NDMI`
  4) `MSAVI`
  5) `VV`
  6) `VH`
  7) `VVVH`
  8) `VVDIFFVH`
- 其中：
  1) `NDVI/EVI/MSAVI` 侧重植被活力；
  2) `NDMI` 侧重植被/地表水分；
  3) `VV/VH/VVVH/VVDIFFVH` 侧重结构与散射差异。
- 当前 `NDWI` 与 `RVI` 也可保留做对照，但建议先不要一次把所有特征都塞进第一版训练，可先做 8 通道版本，再视结果追加。

#### 十、当前最推荐的明日执行顺序
- 明天建议按以下顺序推进：
  1) 先确定要下载的月份范围（建议先 2021–2025 年每年 4–9 月）；
  2) 运行新的 `gee_downloader.py` 批量导出；
  3) 整理导出的月尺度 TIFF，并生成连续月份 `X_tensor`；
  4) 用新标签脚本按指定月份生成时间序列标签；
  5) 用 `build_forecast_v2_dataset.py` 生成真正的 forecasting V2 训练样本；
  6) 最后再考虑让 `forecast_main.py` 读取新的 `forecast_v2_X/Y`。

#### 十一、当前阶段的总建议
- 在数据源受限为 S1/S2 的情况下，forecasting V2 依然完全可以推进；
- 当前最值得优先做的不是继续调模型，而是：
  1) 下载更多月份；
  2) 增加更多 S1/S2 指数；
  3) 让标签支持指定月份；
  4) 建立滑动窗口版 forecasting V2 数据集。
- 本次已完成的实际代码层修改包括：
  1) 重写 `gee_downloader.py`；
  2) 新增 `generate_monthly_threshold_labels.py`；
  3) 新增 `build_forecast_v2_dataset.py`。
- 当前会话内容已按要求详细保存在日志中，方便后续继续衔接。

### 2026-04-20（预测链路已跑通后的下一步建议）
- 用户说明：目前已经完成了预测链路的数据获取、数据预处理与预测模型运行。
- 当前可查看位置：
  1) 过程与中间结果：`/root/autodl-tmp/code/process.ipynb`
  2) 预测模型运行与结果查看：`/root/autodl-tmp/code/forecast-model.ipynb`
  3) 预测结果输出目录：`/root/autodl-tmp/zyk_drought_monitor/results/forecast_compare_V2`
- 当前结果目录中已包含：
  1) `forecast_model_summary.txt`
  2) `forecast_metrics_comparison.png`
  3) `forecast_confusion_matrices_normalized.png`
  4) `forecast_spatial_prediction_compare.png`

#### 一、当前阶段的总体判断
- 结论：当前最优先的工作不应再是继续盲目换模型，而应先完成“预测任务闭环验证 + 误差分析 + 数据定义校正”。
- 主要原因：
  1) 数据获取、预处理、训练、结果导出这条链路已经跑通，说明现阶段的瓶颈不再是“能不能跑起来”；
  2) 结合此前记录，数据链路中仍存在潜在风险，包括：部分月份数据缺失/异常、通道定义与标签规则可能不一致、forecasting 样本构造仍需严格核对；
  3) 当前预测结果并不算理想，说明下一步更需要先判断问题来自“任务定义/数据质量/标签策略”，还是来自“模型本身”。

#### 二、为什么此时不建议立刻继续加模型复杂度
- 当前预测结果摘要显示：
  1) Accuracy 约为 `0.5659`
  2) Macro-F1 约为 `0.3912`
  3) `重/特旱` 类别 support 为 `0`
- 这说明至少存在以下一种或多种情况：
  1) 类别分布明显失衡；
  2) 标签构造规则导致极端类别几乎未出现；
  3) 测试年份本身缺少重旱样本；
  4) forecasting 任务定义与真实未来预测目标之间仍有偏差。
- 因此，此时如果直接继续叠加 attention、替换网络或继续训练更多轮次，得到的提升未必具有解释意义。

#### 三、当前最应该优先做的事情：结果可信性检查
- 建议优先检查以下内容：
  1) `X` 的输入 4 个时间步是否严格按月份顺序组织；
  2) `Y` 是否确实对应第 5 个月，而不是时间错位或标签对齐错误；
  3) 同一 patch 在 `X` 与 `Y` 中的空间位置是否完全一致；
  4) train / val / test 的年份切分中是否存在样本泄漏；
  5) 是否有缺失月、异常月、空影像月份被误纳入 forecasting 样本构造。
- 结论：只有先确认“任务定义是对的”，后续指标变化才有分析价值。

#### 四、下一步必须补做的类别分布与标签统计
- 建议尽快新增统计分析，至少覆盖以下维度：
  1) 每个年份的类别像元数分布；
  2) 每个月份的类别像元数分布；
  3) 每个 patch 的主导类别占比；
  4) train / val / test 各自的类别比例；
  5) 是否存在某些类别在训练集或测试集中几乎缺失。
- 这样做的目的：
  1) 解释为什么 Macro-F1 偏低；
  2) 判断 `重/特旱` support 为 0 是数据事实还是标签构造问题；
  3) 为后续是否合并类别、重设阈值或改标签方案提供依据。

#### 五、当前最值得做的误差分析
- 建议不要只停留在总指标，而要围绕现有结果图做误差拆解：
  1) 查看 confusion matrix 中 `无旱/轻旱/中旱` 是否存在明显互相混淆；
  2) 查看空间预测图，分析是否存在大面积平滑、边界模糊或局部噪声斑块；
  3) 从结果中挑选若干典型样本，分别分析“预测较好”“轻中旱混淆严重”“空间结构明显失真”三类案例；
  4) 判断错误更像是“时序预测失败”，还是“标签阈值划分不合理”。
- 这一步的价值：后续无论是改数据、改标签还是改模型，都能有明确方向，而不是盲调。

#### 六、需要重新核对的数据与标签定义
- 结合此前链路复盘，当前最需要再次确认的是：
  1) 当前 forecasting 特征通道的真实含义是否已经与标签生成脚本中的通道假设完全一致；
  2) 数据下载阶段发生过异常的月份，后续是否被替换、跳过、填补或错误保留；
  3) 当前 100m 分辨率是否已经足够支撑当前尺度的 forecasting 任务。
- 当前建议：
  1) 先把“每个输入通道是什么、标签依赖哪些通道/规则”整理成明确表格；
  2) 再回查缺失月、异常月是否进入训练样本；
  3) 若后续资源允许，可做小范围 100m vs 30m/20m 的对照实验，确认分辨率是否是限制因素之一。

#### 七、下一步应先补简单 baseline，再谈更复杂模型
- 建议优先补两个基线：
  1) Persistence baseline：直接用第 4 个月标签近似预测第 5 个月；
  2) Single-month baseline：只用最近 1 个月特征预测目标月。
- 这两个 baseline 的意义：
  1) 判断深度时序模型是否真的优于“惯性延续”；
  2) 判断多月时序输入是否真的比单月输入带来增益；
  3) 如果深度模型无法稳定超过简单 baseline，则应优先回头检查数据和标签，而不是继续堆模型结构。

#### 八、建议的下一轮实验顺序
- 推荐按“小步可解释”的顺序推进：
  1) 先做标签策略对比：`threshold` vs `kmeans`；
  2) 再做输入时间长度对比：2 个月、3 个月、4 个月；
  3) 再做特征组合消融：如只用 NDVI、NDVI+VV+VH、全特征；
  4) 在上述结论较清楚后，再比较 ConvLSTM / ConvGRU / TrajGRU / attention 版本。
- 结论：模型对比应放在数据定义、标签策略、时间窗口与特征组合之后，否则实验解释力不足。

#### 九、当前最推荐的实际执行顺序
- 建议按下面顺序继续推进：
  1) 统计 forecasting V2 数据集的类别分布、年月分布和 patch 分布；
  2) 核对 `forecast_v2_X` 与 `forecast_v2_Y` 的时间对齐与空间对齐；
  3) 回查异常月份、缺失月份是否被纳入样本；
  4) 基于现有结果图做 confusion matrix + spatial case 的误差分析；
  5) 建立 persistence baseline；
  6) 再开始标签策略、时间窗口长度与特征消融实验；
  7) 最后再进入模型结构层面的进一步比较。

#### 十、当前阶段的一句话结论
- 当前最值得优先做的，不是继续训练更多模型，而是先验证 forecasting 任务与标签是否定义正确、样本是否可靠、误差主要来自哪里；在此基础上，再进入下一轮有解释力的实验设计。

