# Harness 接入任务记录

**任务**：把 `contest_harness` SDK 接入本项目，按平台要求导出单个 Linux/amd64 的
未压缩 Docker save tar，并在平台试运行前完成本地离线验证。

**状态**：完成。镜像已构建并通过校验，离线 30 轮演练全部跑通，124 项测试通过。

> **2026-11-17 修订**：上一版镜像在平台**试运行**被拒（`failure_kind: platform`，
> "平台尚未完成此提交的运行适配，训练未开始"）。根因是该镜像同时携带了四套
> 候选入口，平台无法判定哪一套才是 v2 提交。整改见
> **§3.5 镜像入口歧义（本次试运行被拒的根因）**。
> 本文档中的哈希、镜像内容清单与环境变量**均已按新镜像更新**。

---

## 1. 交付物

| 文件 | 大小 | SHA256 |
| --- | --- | --- |
| `zhixun-engine-linux-amd64.tar` | 174.3 MiB | `5fb7ef173e6df951396201d285a3220e355c5c59ae407cd4f2810e713a499c3f` |
| `dist/zhixun-engine-linux-amd64.tar` | 同上（同一文件） | 同上 |
| `zhixun-engine-linux-amd64.tar.sha256` | 校验和旁挂文件 | — |

镜像属性（`--verify-only` 校验通过）：

```
image tags: ['zhixun-engine-b:latest']
platform: linux/amd64
layers: 6
entry point: python3 /app/harness.py -> contest_submission.py (A/B/C, one model call in training, no shadowing package)
OK  a single uncompressed linux/amd64 docker archive
```

- **未压缩**：每个 layer 都是裸 `layer.tar`，不是 gzip/zstd。
- **单个镜像**：`manifest.json` 里只有一条记录。
- **无模型权重**：归档内 `.npz/.npy/.pt/.pth/.bin/.safetensors/.onnx/.pkl` 等一律为零。
- **入口**：`Entrypoint = ["python3", "/app/harness.py"]`，`Cmd = ["describe"]`。
- **非 root**：`User = 10001:10001`，`WorkingDir = /app`。
- **唯一入口**：`/app` 下只有 `harness.py` 与 `contest_submission.py` 两个可执行入口，
  **没有任何包目录遮蔽它们**（详见 §3.5）。

## 2. 镜像内容

```
/app/harness.py                       SDK 入口模板
/app/contest_submission.py            train(rnd) / predict(sample, model_dir)
/app/contest_baselines/model.py       contest.cpu-mlp.v1 参考实现 + 权重迁移
/app/contest_baselines/data.py        公开投影读取 + 单位分组
/app/contest_baselines/metrics.py     Macro-F1 / RMSE / MAE / NASA 非对称分
/app/contest_baselines/strategies.py  假设库与每轮配置
/app/contest_baselines/train.py       一轮完整闭环
/app/contest_baselines/predict.py     单样本预测
/app/contest_baselines/offline_host.py 本地协议宿主（离线自检用）
/app/contest_harness_sdk/             SDK 源码
/usr/local/lib/python3.12/site-packages/contest_harness/   SDK 1.0.2
/usr/local/lib/python3.12/site-packages/numpy/             numpy 2.3.0
```

镜像里**只有提交，没有实验室**：研究运行时（`harness/`、`rulkit/`、`tracks/`、
`research/`、`evo/`）**不打进镜像**，只留在本仓库与源码链接里。原因见 §3.5。

环境变量：`CONTEST_WORK=/workspace`、`CONTEST_APP_ROOT=/app`、
`RUL_DATA_ROOT=/data`、`PYTHONUTF8=1`、`PYTHONUNBUFFERED=1`。

挂载点目录（平台挂载前必须存在）：`/data`、`/previous`、`/state`、`/model`、
`/workspace`、`/out`、`/tmp`。

**构建工具不进镜像**：`tools/build_docker_image.py` 带拉取 registry 的代码，
属于开发工具，已由 `EXCLUDED_FROM_IMAGE` 排除。

## 3. 本轮修复的真实缺陷

离线演练跑满 30 轮时暴露了四个会真实影响得分的问题，全部已修并加了回归测试。

### 3.1 `resize()` 的 Adam 矩形状错误（`contest_baselines/model.py`）

迁移到新隐层宽度时，Adam 的一阶/二阶矩是按**旧**形状分配的：

```python
first = {k: np.zeros_like(v) for k, v in self.params.items()}   # 旧形状
```

于是 `validate()` 抛 `checkpoint tensor shape or values invalid`，
`capacity-is-overfitting` 这一轮直接崩。改为按**新**形状分配再切片搬运。

同时把 `epochs` / `steps` 计数器**改为保留**而非清零：
epoch 计数驱动洗牌置换、清零会重放父模型已见过的洗牌；
step 计数驱动 Adam 的偏差修正、清零会用错误的分母重新缩放继承下来的矩。

### 3.2 `unit_of()` 只认 Profile B 的 ID 格式（`contest_baselines/data.py`）

原实现只匹配 `_s\d+$`，Profile C 的 `fd001_train_u0001_c00001` 不匹配，
于是**每一行都成了自己的"单位"**：11 万行 → 11 万个组，
`_unit_shortness` 退化成平方级循环，C 数据集第 3 轮直接卡死。

改为匹配通用的 `_[A-Za-z]\d+$`（B 的 `_s`、C 的 `_c` 都覆盖），
纯数字尾缀（A 的 `gas_b01_000001`）不剥离。C 的分组从 113089 降到 496。

### 3.3 `the-unit-is-the-confounder` 改变了特征 schema（`contest_baselines/strategies.py`）

这是**最严重**的一个：它给训练矩阵加了一列 `meta/elapsed`，并写进
`model.meta["paths"]`。但平台的预测请求只有 `{task, inputs}`，**没有 sample id**：

```python
# contest_harness/predict.py
sample = Sample(request["task"], request["inputs"], context)
```

带这列的候选在隐藏集上**每一行都会预测失败**，得分归零；
而且参考加载器本身就会因为 schema 不一致而拒绝该 checkpoint。

改法：同一个假设换一条**不依赖 id、因而可在打分时复现**的通道来检验——
让每个单位在损失中占据相同的总质量（`uniform mass per unit`），
从而把"总体平均寿命"从目标函数里去掉。schema 不变，前向不变。

并在 `prepare()` 末尾加了**硬闸门**：任何改变 `model.meta["paths"]`
或矩阵宽度的改动都会立即抛错，不会等到隐藏评测才发现。

### 3.4 第 6–10 轮退化成重复实验（`contest_baselines/strategies.py`）

epoch 预算 `count += min(round_index - 1, 5) * 5` 在第 6 轮之后封顶，
于是第 6–10 轮跑同一个假设、同一个预算、同一个父模型，**产出完全相同的数字**。
重复实验不算实验。改为严格递增 `count += (round_index - 1) * 10`，上限 150。

### 3.5 镜像入口歧义（本次试运行被拒的根因）

**症状**：平台试运行返回 `failure_kind: platform`，原因写的是
"平台尚未完成此提交的运行适配，训练未开始"。平台的自动分析给出了它看到的东西：

> No faithful single v2 entry serves A,B,C with the required training model call.
> … harness is the ONLY model-calling loop (GET /models + POST /chat/completions,
> wire-compatible with model_transport) but is structurally two-stage bearing-RUL
> (predict_rul, LifeFractionModel, rmse/mae); serving A classification or C needs
> an algorithm rewrite. Documented python -m evo serve/predict is absent from the image.

**根因**：平台审计的对象是**镜像里的代码**，而上一版镜像同时携带了四套并存的
候选入口（`harness/`、`rulkit/`、`tracks/`、`research/`）。三个具体机制叠加：

1. **包目录遮蔽同名模块。** 镜像里 `/app/harness.py`（SDK 入口）与 `/app/harness/`
   （研究包）并存。CPython 的 `FileFinder` **先找包目录、再找同名模块**，所以
   `import harness` 命中的是 `harness/__init__.py`。任何按名字扫描镜像的检查
   都看不到真正的 SDK 入口。（已实测确认：同目录下 `harness/` 胜出。）
2. **唯一走 HTTP 的模型调用循环在研究代码里。** `harness/llm.py` 是镜像内**唯一**
   实现 `GET /models` + `POST /chat/completions` 的代码，而 `harness/experiments.py`
   又是两段式轴承 RUL（`predict_rul` / `LifeFractionModel` / rmse / mae）。
   平台找"会调模型的循环"时找到的就是它，于是认定提交**只做回归**，
   "serving A classification or C needs an algorithm rewrite"。
3. **文档把早期工具写成平台入口。** `docs/系统运行说明.md` 等文档把
   `python -m evo serve` / `python -m evo predict` 描述为评测入口，
   而镜像里**根本没有 `evo`**，于是又一条 "absent from the image"。

**为什么真正的 v2 路径没被发现**：提交的实现（`contest_submission.py` →
`contest_baselines/`）本身是好的——它同时支持分类与回归，训练里调
`rnd.model(...)`，且能读父检查点续训。它只是**没有成为镜像里唯一的候选**，
所以平台的审计结论把研究代码当成了提交。

**改法**：

| 改动 | 文件 |
| --- | --- |
| 镜像只打包提交：`contest_baselines/` + `contest_harness_sdk/` + 四个入口文件 | `tools/build_docker_image.py` |
| 构建期断言：入口齐全，且**没有包遮蔽入口** | `_assert_single_entry()` |
| 校验期断言：入口、`Entrypoint`、A/B/C 覆盖、训练含 `rnd.model(` | `_verify_entry_point()` |
| 新工具：**从归档本身解出 `/app`** 再实跑 `describe` + A/B/C 的 train + predict | `tools/verify_image_roundtrip.py` |
| 文档区分"提交入口"与"研究运行时" | `README.md`、`docs/系统运行说明.md`、`docs/数据适配说明.md`、`docs/训练凭证.md` |
| 根 `Dockerfile` 改为**提交镜像**；原研究镜像移为 `Dockerfile.research`；删除重复的 `Dockerfile.harness` | 仓库根 |
| CI 增加"单入口检查"与"describe"两个步骤 | `.github/workflows/ci.yml` |

**验证结果**（新镜像，`fd5fe06f…`）：

```
extracted 39 file(s) from the archive's /app
OK    layout: one entry point, no shadowing package, no research code
OK    describe: {"protocol": "contest.harness.workspace.v2", "model_access": "host_gateway_rpc", "sdk": "contest_harness", "sdk_version": "1.0.2", "tasks": ["classification", "regression"]}
OK    A train: [contest] candidate macro_f1 = 0.7192 (30 epochs, 0.5s) | 1 model call(s) | candidate submitted
OK    A predict: classification -> 6 labels
OK    B train: [contest] candidate rmse = 663.9085 (30 epochs, 1.5s) | 1 model call(s) | candidate submitted
OK    B predict: regression -> value=588.6843
OK    C train: [contest] candidate rmse = 51.3511 (30 epochs, 14.3s) | 1 model call(s) | candidate submitted
OK    C predict: regression -> value=181.151
```

A 方向的父模型 macro-F1 0.7140 → 候选 0.7192；B 方向 RMSE 681.16 → 663.91；
C 方向 RMSE 52.87 → 51.35。三个方向**同一套代码**，每轮各有一次真实模型调用。

### 3.6 模型调用返回空正文（第二版试运行失败的根因）

**症状**：第二版镜像（`a5474acf…`）能上传、能通过校验，但试运行报
「**模型服务暂不可用，请稍后重试**」。

**排除项**：

- 不是余额问题：同一个 Key 连打 6 次 `Qwen/Qwen3.5-4B`，**全部 HTTP 200**
  （首次 5.4 s，之后 0.2 s），`usage` 正常计费。零余额返回的是
  HTTP 409 `insufficient_quota`，与本症状不同。
- 不是 PyTRiO 服务不可用：同上，6/6 成功。
- 不是超时分类：平台前端的失败分类表里没有这一条文案，
  说明它是**后端返回的原始错误**，不是 `TIMEOUT` / `BUDGET` /
  `CONTROL_HTTP_` 中的任何一类。

**真因**：平台的模型是**推理模型**，默认开思维链。用训练循环真实发出的
那条 prompt 实测（`max_tokens=512`，即原代码的取值）：

| 模型 | 思维链 | 延迟 | `finish_reason` | `content` |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | 开 | 3.8 s | `length` | **空** |
| `Qwen/Qwen3.8-27B` | 开 | **288.3 s** | `length` | **空** |
| `Qwen/Qwen3.5-4B` | 关 | 9.0 s | `stop`/`length` | 842 字符 |
| `Qwen/Qwen3.8-27B` | 关 | 12.0 s | `stop` | 793 字符 |

思维链吃掉了全部 token 预算，正文为空；27B 还要 288 秒，网关等不到就报
「模型服务暂不可用」。**这个症状与"模型大小"的关系是反的**：27B 更慢，
更容易触发。

**改法**（`contest_baselines/train.py`）：

| 改动 | 说明 |
| --- | --- |
| `enable_thinking=False` | 关掉思维链，正文直接返回 |
| token 预算 512 → 1024 | `_MODEL_TOKEN_BUDGET`，给正文留够空间 |
| 重试回退（3 次，间隔 3/6 s） | 服务偶发不可用或返回空正文时不丢弃已完成的测量 |

`enable_thinking` 是官方 SDK 白名单里的合法参数
（`contest_harness/round.py::_MODEL_OPTIONS`）；宿主不认识时，
重试会退回不带该参数，所以不会导致调用被拒。

**回归测试**（`tests/test_model_call_fallback.py`）用宿主演示替身
制造"空正文"和"调用报错"两种故障，断言轮次仍然完成并带出模型调用号。

### 3.7 SDK 版本对齐

第二版镜像的 `describe` 报 `sdk_version: 1.0.1`，平台文档写的是 `1.0.2`。
原因不在镜像里——镜像打包的 `contest_harness_sdk/` 一直是 1.0.2——而是
**本地 `.venv` 里装的是旧的 1.0.1**，离线测试实际跑的是旧版。
1.0.1 与 1.0.2 的 `transport.py` 有一处实质差异（流式 `finish_reason`
的兜底值），这会让本地验证与正式运行存在偏差。

已把 `.venv` 重新安装为 1.0.2，本地测试从此与镜像跑同一份代码。

## 4. 续训是否真的是"续训"

| 检查项 | 结论 |
| --- | --- |
| 父检查点来源 | `rnd.parent.path("checkpoint.npz")`，走 SDK 的 digest 校验 |
| 权重 | 从父检查点原样载入，候选 = 父检查点的精确副本再训练 |
| Adam 矩 | 一阶/二阶矩从父检查点继承，迁移时按存活单元搬运 |
| epoch / step 计数 | 继承，不清零 |
| 归一化状态 | `mean` / `scale` 继承，不重算 |
| 目标语义 | `target_mean` / `target_scale` 继承；直接拟合时写 1.0/0.0，参考加载器还原出的数值完全相同 |
| 特征 schema | 继承，硬闸门保护 |
| 格式 | `contest.cpu-mlp.v1`：`w1/b1/w2/b2` + `adam_first_*` + `adam_second_*` + `mean` + `scale` + `metadata` |

## 5. 密钥与外联

`tools/scan_for_secrets.py` 分两层扫描：

```
== the runtime package (what the platform executes) ==
scanned 10 file(s): contest_submission.py, harness.py, contest_baselines
OK  the runtime package carries no credential, no private key and no external model client
```

模型访问走平台宿主的 `model_access: host_gateway_rpc`，
调用方式是 `rnd.model(messages)`，镜像内**没有**任何密钥、私钥、
外部模型客户端或出站 HTTP 客户端。

## 6. 测试结果

```
198 passed, 1 warning, 100 subtests passed in 35.30s      # 本项目全部测试
Ran 15 tests in 9.526s  OK                                 # SDK 1.0.2 自带测试
```

新增的三组回归测试：

| 文件 | 覆盖 |
| --- | --- |
| `tests/test_migration.py` | 迁移后的形状、矩的搬运、计数继承、可训练性、可往返 |
| `tests/test_data_projection.py` | 三个 profile 的单位分组、展平遍历顺序、禁用的目标派生字段 |
| `tests/test_strategies.py` | schema 不变闸门、每个假设可测、预算严格递增、计划不重复 |

## 7. 离线 30 轮演练结果

```bash
python tools/run_harness_offline.py --rounds 10 --live --out runs/harness-offline
```

演练用一个本地 host 替身走**同一套 JSONL 协议**，跑的是选手自己的
`train`/`predict`，用**真实公开数据包**和**真实父检查点**。

> **`--live` 用 PyTRiO 的真实 27B 驱动决策层**（`Qwen/Qwen3.8-27B`，
> 思维链关闭），不加该参数时由本地替身作答。
> 报告里的 `live` 字段和每轮的 `live_model` 字段写明那一轮到底是哪一种，
> 读者不必猜。**两种模式证明的都是代码路径，不是分数**——
> 演练用的是公开验证集，隐藏集上的表现仍以平台测量为准。

| 数据集 | 轮次 | 假设 | 采纳 | 主指标 |
| --- | --- | --- | --- | --- |
| A | 1 | more-data-helps | 是 | macro_f1 0.7140 → 0.7192 |
| A | 2 | the-parent-was-left-unconverged | 否 | 0.7192 → 0.6157 |
| A | 3 | the-scale-is-swamping-the-signal | 是 | 0.7192 → 0.7358 |
| A | 4 | the-rare-class-needs-the-loss-to-notice-it | 是 | 0.7358 → 0.7371 |
| A | 5 | capacity-is-overfitting | 否 | 0.7371 → 0.7370 |
| A | 6 | rare-class-is-underweighted | 否 | 0.7371 → 0.7367 |
| A | 7 | the-batch-is-mixing-scales | 是 | 0.7371 → 0.7391 |
| A | 8 | more-data-helps | 是 | 0.7391 → 0.7400 |
| A | 9 | more-data-helps | 是 | 0.7400 → 0.7405 |
| A | 10 | more-data-helps | 是 | 0.7405 → 0.7410 |
| B | 1 | more-data-helps | 是 | rmse 681.16 → 663.91 |
| B | 2 | the-parent-was-left-unconverged | 是 | 663.91 → 621.26 |
| B | 3 | the-tail-is-underweighted | 否 | 621.26 → 672.71 |
| B | 4 | the-unit-is-the-confounder | 是 | 621.26 → 591.27 |
| B | 5 | capacity-is-overfitting | 是 | 591.27 → 578.67 |
| B | 6 | more-data-helps | 是 | 578.67 → 547.32 |
| B | 7 | more-data-helps | 是 | 547.32 → 537.49 |
| B | 8 | more-data-helps | 是 | 537.49 → 529.80 |
| B | 9 | more-data-helps | 是 | 529.80 → 490.25 |
| B | 10 | more-data-helps | 是 | 490.25 → 456.42 |
| C | 1 | more-data-helps | 是 | rmse 52.87 → 52.23 |
| C | 2 | the-parent-was-left-unconverged | 否 | 52.23 → 51.24 |
| C | 3 | the-tail-is-underweighted | 否 | 切片 38.52 → 34.52，**全量 +7.31% 被守门否决** |
| C | 4 | the-unit-is-the-confounder | 是 | 38.52 → 36.15 |
| C | 5 | capacity-is-overfitting | 否 | 36.15 → 39.01 |
| C | 6–10 | more-data-helps | 否 | 切片改善但全量守门不过，父模型保留 |

**C 第 3 轮是这套判据存在的理由**：切片指标从 38.52 掉到 34.52
（**−10.4%**，看起来是本轮最大的一次提升），但**全量**指标同时劣化了
**+7.31%**，超出 5% 的守门容差，所以候选被否决、父模型保留。

这正是平台 2026-11-22 试运行踩到的坑：当时上线的候选在隐藏集上
**56.253**，比 52.8686 的静态基线**差 6.40%**，违反平台
「效果不低于平台基线」的硬性要求——而公开切分**低估**了隐藏难度
（51.111 公开 vs 56.253 隐藏，差 +10.06%）。
只看全量会漏判，只看切片也会误判，**两者并列**才拦得住。

产物：`runs/harness-offline/run_report.json`、`runs/harness-offline/ledger.jsonl`，
以及 `runs/harness-offline/swanlab/<数据集>-round-<n>.jsonl`（逐轮实验记录）。

## 8. 平台操作步骤

1. 在 `https://rsi-index.115.zone/#submissions` 上传
   `zhixun-engine-linux-amd64.tar`（174.3 MiB 的**未压缩** tar，不要打包成 zip）。
   SHA256：`5fb7ef173e6df951396201d285a3220e355c5c59ae407cd4f2810e713a499c3f`。
   > **必须上传新包**：上一版（`9e6c1415…`）在试运行时被拒，根因见 §3.5；
   > 再上一版（`a5474acf…`）能上传但试运行报「模型服务暂不可用」，根因见 §3.6；
   > 第三版（`fd5fe06f…`）能跑完，但试运行的隐藏分 **56.253** 比基线
   > **差 6.40%**，违反「效果不低于平台基线」，根因见 §7 与 §8。
2. 平台会依次执行 `describe` → `train` × 10 → `predict`。
   入口声明通过**不代表**训练、预测和比赛要求都已通过。
3. **模型选 `Qwen/Qwen3.8-27B`**（赛题白名单内，见
   `docs/平台提交表单填写指南.md` §2.2）：
   分数由工业模型决定，LLM 大小不进入任何分数项；
   **思维链必须关掉**（`enable_thinking=False`）——
   上表实测：27B 关思维链只要 **12.0 s** 且正文正常返回（793 字符），
   开思维链则 288.3 s 且正文为空。
   > 旧版本文档曾建议改用 `Qwen/Qwen3.5-4B`，理由是"27B 开思维链要 288 秒"。
   > 那个前提已经被 `enable_thinking=False` 消掉了——**关掉思维链后 27B 比 4B 还快**
   > （12.0 s vs 9.0 s，且 27B 正文更完整）。改用 4B 不是必需，且等于放弃了更大的模型。
4. 若要本地复核：

```powershell
python tools/build_docker_image.py --verify-only --out zhixun-engine-linux-amd64.tar
python tools/verify_image_roundtrip.py --archive zhixun-engine-linux-amd64.tar
python -m unittest discover -s tests -t . -v
python tools/scan_for_secrets.py
python tools/run_harness_offline.py --rounds 10 --live --out runs/harness-offline
```

**表单每个字段填什么**，见 [`docs/平台提交表单填写指南.md`](平台提交表单填写指南.md)。
其中「完整 Sampler 权重路径」一栏的 `trio://` 路径来自 PyTRiO 的
`save_weights_for_sampler(...).result()`，**不在本仓库也不在镜像里**；
没有微调权重时应选「基础模型（需二次确认，可正式参赛）」。

## 9. 已知限制

- 离线演练可以用**真实模型**驱动，也可以由本地替身作答。
  默认是替身；加 `--live` 则通过 PyTRiO 打真实模型
  （`HARNESS_LLM_BASE_URL` / `HARNESS_LLM_API_KEY` / `HARNESS_LLM_MODEL`）。
  无论哪种，**都不能**用来预测平台上的真实分数——
  演练用的是公开验证集，隐藏集上的表现仍以平台测量为准。
- 演练的报告里 `live` 字段与每轮的 `live_model` 字段会写明那一轮
  到底是**真模型**还是**替身**作答的，不需要读者猜。
- 第 6–10 轮在假设穷尽后会重复最优假设。严格递增的 epoch 预算让重复仍是
  不同的实验，但一个真正穷尽后应当做的动作是"把已验证的结论固化下来"，
  而不是再跑一遍同样的配置——这一点留给平台上的真实轮次观察。
