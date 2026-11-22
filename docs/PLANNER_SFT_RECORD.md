# 决策层 27B LoRA 微调记录（PyTRIO）

> 记录日期：2026-11-21
> 基座：`Qwen/Qwen3.8-27B`（PyTRIO 平台模型 ID）
> 用途：Harness 决策层（诊断卡 → 结构化动作 JSON），不是工业模型本身。
> 本文件不含 API Key；Key 只填在平台表单的密钥栏。

## 1. 产物

| 项 | 值 |
|---|---|
| Sampler 权重路径 | `trio://run_7yqgoj5be2n0/sampler_weights/planner-27b-sft-v1` |
| 训练配置 | LoRA rank 32，lr 1e-4，1 epoch，batch 1，max_seq_len 8192 |
| 训练样本 | 233 条（9 条真实决策 + 100 条策略合成 + 180 条平衡反事实） |
| 留出样本 | 56 条 |
| 训练耗时 | 1285.5 s（233 步，含 tokenizer 下载） |
| 逐步 loss | `train/reports/planner-27b-sft-v1.log`（本次标准输出副本）；脚本现在同时写 `.steps.jsonl` |
| 最终训练 loss | 末段 0.00002–0.10；最终报告 `train/reports/planner-27b-sft-v1.json` |
| 训练脚本 | `train/sft_planner.py` |
| 数据脚本 | `train/build_planner_sft.py` |
| 评测脚本 | `train/eval_planner.py` |
| 评测报告 | `train/reports/eval-{base-full,sft-full,base-real,sft-real}.json` |

## 2. 数据来源

全部来自本仓库：

| 来源 | 数量 | 说明 |
|---|---:|---|
| `evidence/ledger.jsonl` + `evidence/hypothesis_chain.jsonl` | 9 | 真实决策：诊断卡、动作、假设、参数、门控结果 |
| `harness/policy.py` 合成 | 100 | 在真实字段空间上采样诊断卡，由确定性策略打标签 |
| 平衡反事实 | 180 | 6 个动作各 30 条，补齐策略从不输出的 `adapt` / `rollback` |

训练 prompt 由 `harness.llm.SYSTEM_PROMPT`、`USER_TEMPLATE` 和
`compact_history()` 直接渲染，`harness/skills/*.md` 按运行时同样顺序拼接；
训练与推理不会出现 prompt 漂移。

数据边界：合成样本是本地策略/人工反事实，不是线上测量结果；9 条真实决策
才是金标准。训练报告按 `real` / `policy` / `balanced` 分开统计。

## 3. 评测

完整 56 条留出卡：

| 指标 | 基座 27B | SFT 27B |
|---|---:|---:|
| JSON 解析率 | 100% | **100%** |
| schema 合法率 | 100% | **100%** |
| 动作一致率 | 69.6% | **94.6%** |
| 平均延迟 | 4.1 s | **3.0 s** |
| 平均输出字符 | 514.8 | **245.4** |
| 覆盖动作数 | 5 / 6 | **6 / 6** |

真实 9 条决策卡（训练集内，仅作复现性检查）：动作一致率
66.7% → **88.9%**，延迟 3.8 s → **2.7 s**。

真实 B 方向闭环（4 线程，`stream_mode=bearing_holdout`）：

| 决策层 | 墙钟 | 峰值 RSS | 模型调用 | 最终 RMSE |
|---|---:|---:|---:|---:|
| 确定性策略 | 123.3 s | 80.9 MB | 0 | 706.4 |
| SFT 27B | 170.7 s | 84.5 MB | 11 次 / 20,966 tokens | 706.4 |

前 11 个周期 `planner=qwen`；第 12 个周期起 20,000 token 预算用尽，
按设计自动回落确定性策略。两次最终 RMSE 相同，因为实验预算在第 3 个周期
后已经用尽，决策层无法改变「预算耗尽 → do_nothing」的硬约束。
706.4 低于基线 681.16（−3.71%）是这一轮 3 根轴承切分的负结果，与仓库
文档记录的「官方切分 −5.2%」一致；9 折留一轴承的 +48.86% 是另一套协议，
本次没有重跑。

## 4. 复现

```bash
# 1) 生成数据（不联网）
python -m train.build_planner_sft --out train/data \
  --synthetic 100 --balanced-per-action 30

# 2) 训练（PyTRIO）
python -m train.sft_planner --data train/data/planner-train.jsonl \
  --epochs 1 --batch-size 1 --lr 1e-4 --rank 32 \
  --save-name planner-27b-sft-v1

# 3) 评测
python -m train.eval_planner --data train/data/planner-val.jsonl --limit 56 --name base-full
python -m train.eval_planner --data train/data/planner-val.jsonl --limit 56 --name sft-full \
  --model-path 'trio://run_7yqgoj5be2n0/sampler_weights/planner-27b-sft-v1'
```

## 5. 平台表单对应

| 表单字段 | 填写 |
|---|---|
| 模型类型 | 持久化 Sampler 微调权重 |
| 模型 / 权重路径 | `trio://run_7yqgoj5be2n0/sampler_weights/planner-27b-sft-v1` |
| 训练记录链接 | 本文件推送到公开仓库后的 raw HTTPS 地址 |
| 源码仓库链接 | `https://github.com/GY-phd/ai-` |
| PyTRIO API Key | 只在表单密钥栏填；不要写进仓库、镜像或本文档 |

## 6. 已知边界

- 训练数据以本地策略/反事实为主，不是大规模线上轨迹；提升主要体现在
  动作覆盖率和一致性，而不是新策略的发现。
- 27B 不在工业 CPU 本地推理；工业机只发 OpenAI 兼容请求，断网或预算耗尽
  时走 `harness/policy.py`。
- 4 线程 BLAS 会改变浮点归约顺序；需要逐位复现历史产物时设
  `BASELINE_CPU_THREADS=1`。
- 本机没有 Docker，提交镜像的 4 核/8G 容器构建尚未在本会话执行；
  C 方向提交路径已用离线宿主跑通，B 方向完整闭环已用真实数据跑通。
