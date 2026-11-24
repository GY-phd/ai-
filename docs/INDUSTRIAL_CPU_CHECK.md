# 工业级 CPU 运行核查（2026-11-21）

> 结论先说：**工业模型和 Harness 可以跑在 4 核 / 8G 的 CPU 工业机上；
> 27B 决策层不在本地跑，走 PyTRIO 的 OpenAI 兼容端点，断网时自动回落确定性策略。**
> 目前还差一次带 **B 方向真实数据包** 的端到端核查——本机没有
> `cpu-b-rul-20261110-v1`，所以下面用 **C 方向真实数据** 的提交路径做实测，
> 并列出补做 B 方向核查的命令。

## 1. 两条运行路径要分开看

| 路径 | 跑在哪里 | 依赖 | 4 核/8G 是否够 |
|---|---|---|---|
| **工业模型 + Harness**（`rulkit` / `contest_baselines`） | 工业机本地 CPU | NumPy，无 GPU、无网络 | 够，见第 2 节 |
| **27B 决策层** | PyTRIO 云端 / 评测网关 | HTTPS（OpenAI 兼容） | 不在本地跑；断网走 `harness/policy.py` |

提交镜像 `Dockerfile` 里**没有** 27B 权重、没有推理服务、没有 PyTRIO Key；
平台通过 `contest_harness` 的 `rnd.model()` 代持模型访问。工业机只负责
训练/推理那个几千参数的 NumPy MLP。

## 2. 真实数据实测（C 方向，提交路径）

用 `contest_baselines.offline_host` 的离线宿主，数据根指向
`harness-minimal-loop/cpu-c-guide-20261110-v1`，跑
`python harness.py train` 的完整一轮（读父检查点 → 训练候选 → 指标 →
提交候选 → 一次模型调用帧）。数据规模 113,089 训练行 × 24 特征，
15 epochs，hidden 32。

| 线程 | 墙钟 | 子进程峰值 RSS | 候选 RMSE |
|---:|---:|---:|---:|
| 1 | 19.0 s | 240.9 MB | 52.2253 |
| 4 | **16.2 s** | **240.3 MB** | 52.2253 |

同一条命令的输出（4 线程）：

```
selected_artifact_id = offline-artifact-1
validation rmse 52.8686 -> 52.2253, adopted=yes
model call offline-call-1 replied with 103 characters
round 1 finished in 15.0 s
```

对照赛题给 Harness 的资源上限 **4 核 / 8G**：峰值内存约为上限的 **3%**，
单轮训练约为 `wall_seconds=600` 的 **2.7%**。这不是估算，是子进程
`psutil` 采样的峰值。

## 3. B 方向研究闭环的实测

B 数据包现在在 `C:\harness_review\数据源\cpu-b-rul-20261110-v1`
（7,032 训练行、1,679 验证行、34 特征），按仓库原协议
`stream_mode=bearing_holdout`（整根轴承留出）在 4 线程下跑完整闭环：

| 决策层 | 墙钟 | 峰值 RSS | 周期 | 模型调用 | 最终 RMSE |
|---|---:|---:|---:|---:|---:|
| 确定性策略（`--llm-mode policy`） | 123.3 s | 80.9 MB | 16 | 0 | 706.4 |
| SFT 27B（`trio://.../planner-27b-sft-v1`） | 170.7 s | 84.5 MB | 16 | 11 次 / 20,966 tokens | 706.4 |

两次的最终 RMSE 都是 **706.4**，比静态基线 681.16 低 3.71%。这不是 CPU
问题，而是**切分口径**问题：这一轮留出的 3 根轴承（Bearing3_1 / 2_1 / 2_2）
正好落在仓库文档记录的「官方切分（3 根轴承）−5.2%」负结果附近。仓库的
9 折留一轴承协议是 +48.86%，但那是另一套切分，本机没有重跑。
SFT 版比策略版多花的 ~47 秒，就是 11 次 27B 调用的成本；token 预算用尽后
自动回落策略，闭环照常跑完。

复现：

```bash
export RUL_DATA_ROOT="C:/harness_review/数据源/cpu-b-rul-20261110-v1"
export BASELINE_CPU_THREADS=4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

python -m harness.run \
  --config tmp/verify-b-config.json \
  --llm-mode policy
```

本机**没有 Docker**，所以 4 核 / 8G 容器的镜像构建还没在本会话执行；
第 2 节的 C 方向提交路径已经用同一套离线宿主跑通。

### 3.1 Docker 4 核/8GB 容器实测

Docker Desktop 4.99 + WSL 3.0.1，WSL VM 6 vCPU / 10.4 GB；
Docker Hub 直连不可达，配了 registry mirror
`https://docker.1panel.live`；PyPI 走清华镜像（`PIP_INDEX_URL` 构建参数）。

| 镜像 | 大小 | 命令 | 结果 |
|---|---:|---|---|
| `zhixun-engine-research` | 291 MB | `docker run --rm --cpus=4 --memory=8g --network none ...` | 16 周期、5 次实验、训练 24.74 s、预算墙钟约 28 s、最终 RMSE 706.435、无 OOM |
| `zhixun-engine-b` | 276 MB | `docker run --rm --cpus=4 --memory=8g zhixun-engine-b describe` | 返回 `contest.harness.workspace.v2` 合约，A/B/C 任务声明正确 |
| `zhixun-engine-b` | 276 MB | `OfflineHost` 驱动 `docker run -i --cpus=4 --memory=8g ... train` | 完整一轮 C 方向 train：父 RMSE 52.8686 → 候选 52.2253，adopted=yes，1 次模型调用，容器内 16.8 s，returncode=0 |

容器内 `stream_audit`：6 根初始轴承 / 3 根流轴承 / 3 根验证轴承，
`bearing_leakage=false`、`validation_used_for_training=false`。

构建前修了三处镜像问题：

- `.dockerignore` 排除了 `runs/`，`Dockerfile.research` 仍 `COPY runs/` → 删除该行；
- `tools/verify_repo.py --weights-only` 在无 `.git` 的镜像里误判失败 → 该模式跳过
  git 跟踪集检查，磁盘权重检查保留；
- Docker Hub / `files.pythonhosted.org` 直连不可达 → registry mirror +
  `PIP_INDEX_URL` 构建参数。

复现脚本：`C:\harness_review\docker-verify.ps1`（研究镜像构建 + 4c/8g 闭环 +
提交镜像构建 + describe）。

## 4. 27B 决策层的真实调用

关掉思维链后，用仓库真实诊断卡实测：

```
elapsed_s 4.8   tokens 1767   attempts 1   thinking_off true
action do_nothing   schema valid
```

旧实现（思维链开）是 **288.3 s + 空正文**。工业机不需要本地 27B：
- 有端点：`HARNESS_LLM_BASE_URL` + `HARNESS_LLM_API_KEY` + `HARNESS_LLM_MODEL`
  （后者可填 `trio://.../sampler_weights/...`）；
- 无端点：`--llm-mode policy`，闭环照常跑完，决策记录标 `fallback: true`。

## 5. 结论与剩余风险

**可以确保的：**
- 工业模型是纯 NumPy、几千参数；真实 C 数据 4 线程 16.2 s、峰值 240 MB，
  真实 B 数据完整闭环 4 线程 123.3 s、峰值 80.9 MB（SFT 决策层 170.7 s、
  84.5 MB），都远低于 4 核/8G 限额。
- **Docker 4 核/8GB/无网络容器已实跑**：研究镜像 291 MB，B 闭环 16 周期、
  5 次实验、训练 24.74 s、预算墙钟约 28 s、RMSE 706.435、无 OOM；
  提交镜像 276 MB，`describe` 合约正确。
- Harness 不依赖 GPU/外网；27B 不可达或 token 预算耗尽时确定性策略兜底。
  真实 B 闭环里第 12 个周期起就是因为 20,000 token 预算用尽自动回落的。
- 27B 调用已从“288 s 空正文”修到“4.8 s 合法 JSON”；SFT 后留出 56 卡
  动作一致率 94.6%，真实 9 卡 88.9%。

**还不能拍胸脯的：**
- 容器内已经验证 `describe` 和一轮完整 `train`；`predict` 的完整平台协议
  还没有在容器内驱动，留到平台正式运行时覆盖。
- 4 线程 BLAS 会改变浮点归约顺序；需要逐位复现历史产物时设
  `BASELINE_CPU_THREADS=1`。
- B 方向这一轮 3 根轴承切分的最终 RMSE 706.4 低于基线 681.16（−3.71%），
  与仓库文档记录的“官方切分 −5.2%”一致；这是切分口径的负结果，
  不是 CPU 资源问题。9 折留一轴承的 +48.86% 是另一套协议，本机未重跑。

Docker 已补齐；剩下的是在平台上正式跑一次完整的 train/predict 协议，
容器 `describe` 和研究闭环已经在本机 4 核/8GB 限制下验证过。
