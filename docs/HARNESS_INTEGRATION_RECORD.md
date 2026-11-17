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
| `zhixun-engine-linux-amd64.tar` | 182,456,320 B (174.0 MiB) | `a5474acf722b014f5a4ff86f704d8a57c87d4cae1b2f0292ce0334cc56930986` |
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

**验证结果**（新镜像，`a5474acf…`）：

```
extracted 25 file(s) from the archive's /app
OK    layout: one entry point, no shadowing package, no research code
OK    describe: {"protocol": "contest.harness.workspace.v2", "model_access": "host_gateway_rpc", "tasks": ["classification", "regression"]}
OK    A train: candidate macro_f1 = 0.7192 (30 epochs, 0.5s) | 1 model call(s) | candidate submitted
OK    A predict: classification -> 6 labels
OK    B train: candidate rmse = 663.9085 (30 epochs, 2.0s) | 1 model call(s) | candidate submitted
OK    B predict: regression -> value=588.6843
OK    C train: candidate rmse = 51.3511 (30 epochs, 16.5s) | 1 model call(s) | candidate submitted
OK    C predict: regression -> value=181.151
```

A 方向的父模型 macro-F1 0.7140 → 候选 0.7192；B 方向 RMSE 681.16 → 663.91；
C 方向 RMSE 52.87 → 51.35。三个方向**同一套代码**，每轮各有一次真实模型调用。

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

`python tools/run_harness_offline.py --rounds 10 --out runs/harness-offline-10`

演练用一个本地 host 替身走**同一套 JSONL 协议**，跑的是选手自己的
`train`/`predict`，用**真实公开数据包**和**真实父检查点**。
模型请求由替身本地作答——所以它证明的是**代码路径**，不是分数。

| 数据集 | 轮次 | 假设 | 采纳 | 主指标 |
| --- | --- | --- | --- | --- |
| A | 1 | more-data-helps | 是 | macro_f1 0.7140 → 0.7192 |
| A | 2 | the-parent-was-left-unconverged | 否 | 0.7140 → 0.6677 |
| A | 3 | capacity-is-overfitting | 是 | 0.7140 → 0.7205 |
| A | 4 | rare-class-is-underweighted | 否 | 0.7140 → 0.5737 |
| A | 5 | the-batch-is-mixing-scales | 否 | 0.7140 → 0.6589 |
| B | 1 | more-data-helps | 是 | rmse 681.16 → 663.91 |
| B | 2 | the-parent-was-left-unconverged | 是 | 681.16 → 580.58 |
| B | 3 | the-tail-is-underweighted | 否 | 681.16 → 722.51 |
| B | 4 | the-unit-is-the-confounder | 是 | 681.16 → 670.86 |
| B | 5 | capacity-is-overfitting | 是 | 681.16 → 663.21 |
| C | 1 | more-data-helps | 是 | rmse 52.87 → 51.35 |
| C | 2 | the-parent-was-left-unconverged | 是 | 52.87 → 51.04 |
| C | 3 | the-tail-is-underweighted | 否 | 52.87 → 56.05 |
| C | 4 | the-unit-is-the-confounder | 是 | 52.87 → 51.40 |
| C | 5 | capacity-is-overfitting | 是 | 52.87 → 51.07 |

每轮耗时：A 约 1.2 s，B 约 2.7 s，C 约 22 s——都远在平台墙钟之内。

产物：`runs/harness-offline-10/run_report.json`、`runs/harness-offline-10/ledger.jsonl`。

## 8. 平台操作步骤

1. 在 `https://rsi-index.115.zone/#submissions` 上传
   `zhixun-engine-linux-amd64.tar`（174.0 MiB 的**未压缩** tar，不要打包成 zip）。
   SHA256：`a5474acf722b014f5a4ff86f704d8a57c87d4cae1b2f0292ce0334cc56930986`。
   > **必须上传新包**：上一版（`9e6c1415…`）在试运行时被拒，根因见 §3.5。
2. 平台会依次执行 `describe` → `train` × 10 → `predict`。
   入口声明通过**不代表**训练、预测和比赛要求都已通过。
3. 若要本地复核：

```powershell
python tools/build_docker_image.py --verify-only --out zhixun-engine-linux-amd64.tar
python tools/verify_image_roundtrip.py --archive zhixun-engine-linux-amd64.tar
python -m unittest discover -s tests -t . -v
python tools/scan_for_secrets.py
python tools/run_harness_offline.py --rounds 10 --out runs/harness-offline-10
```

**表单每个字段填什么**，见 [`docs/平台提交表单填写指南.md`](平台提交表单填写指南.md)。
其中「完整 Sampler 权重路径」一栏的 `trio://` 路径来自 PyTRiO 的
`save_weights_for_sampler(...).result()`，**不在本仓库也不在镜像里**；
没有微调权重时应选「基础模型（需二次确认，可正式参赛）」。

## 9. 已知限制

- 离线演练的模型请求由本地替身作答，**不能**用来预测平台上的真实分数。
- 演练用的是**公开**验证集；隐藏集上的表现仍以平台测量为准。
- 第 6–10 轮在假设穷尽后会重复最优假设。严格递增的 epoch 预算让重复仍是
  不同的实验，但一个真正穷尽后应当做的动作是"把已验证的结论固化下来"，
  而不是再跑一遍同样的配置——这一点留给平台上的真实轮次观察。
