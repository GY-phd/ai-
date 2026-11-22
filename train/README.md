# 决策层 27B SFT（PyTRIO）

把 `harness/llm.py` 的决策层从「基座 27B + 规则兜底」训练成
「27B LoRA + 规则兜底」。训练只改决策层，不改工业模型 `rulkit`，
也不改提交镜像的 `contest_baselines`。

## 1. 数据

全部来自本仓库，不依赖 B 方向数据包：

| 来源 | 数量 | 说明 |
|---|---:|---|
| `evidence/ledger.jsonl` + `evidence/hypothesis_chain.jsonl` | 9 | 真实决策：诊断卡、动作、假设、参数、门控结果 |
| `harness/policy.py` 合成 | 100 | 在真实字段空间上采样诊断卡，由确定性策略打标签 |
| 平衡反事实 | 180 | 每种动作 30 条，补齐策略从不输出的 `adapt` / `rollback` |

`train/data/` 下是生成结果：

- `planner-real.jsonl`：9 条真实决策，保留作审计。
- `planner-train.jsonl`：233 条训练样本。
- `planner-val.jsonl`：56 条留出样本（合成部分按 20% 切分）。
- `manifest.json`：来源、动作分布、种子。

每条样本的 `system` / `user` 由 `harness.llm.SYSTEM_PROMPT`、
`USER_TEMPLATE` 和 `compact_history()` 直接渲染，`skills` 也从
`harness/skills/*.md` 按运行时同样的顺序拼接，因此训练 prompt 与推理
prompt 不会漂移。

## 2. 生成数据

```bash
python -m train.build_planner_sft \
  --out train/data --synthetic 100 --balanced-per-action 30
```

## 3. 训练

```bash
# 不联网，只校验 JSONL
python -m train.sft_planner --data train/data/planner-train.jsonl --inspect

# 2 条样本冒烟（约 32 秒，含 tokenizer 下载）
python -m train.sft_planner --data train/data/planner-train.jsonl \
  --limit 2 --epochs 1 --batch-size 1 --save-name planner-smoke

# 正式训练
python -m train.sft_planner --data train/data/planner-train.jsonl \
  --epochs 1 --batch-size 1 --lr 1e-4 --rank 32 \
  --save-name planner-27b-sft-v1
```

训练结束会打印并落盘 `train/reports/<save-name>.json`，关键是
`save_weights_for_sampler()` 返回的完整 `trio://` 路径。平台提交表单选
「持久化 Sampler 微调权重」时原样粘贴这个路径。

## 4. 评测

```bash
# 基座对照
python -m train.eval_planner --data train/data/planner-val.jsonl --limit 20 --name base

# 微调后
python -m train.eval_planner --data train/data/planner-val.jsonl --limit 20 \
  --name sft --model-path 'trio://.../sampler_weights/planner-27b-sft-v1'
```

指标是解析率、schema 合法率、与策略动作的一致率、平均延迟。动作一致率
不是唯一目标：`adapt` / `rollback` 的平衡样本本来就不由策略产生，评测报告
会按来源分别统计。

基座 27B 与 SFT 在完整 56 条留出卡上的对照
（`train/reports/eval-base-full.json`、`eval-sft-full.json`）：

| 指标 | 基座 27B | SFT 27B |
|---|---:|---:|
| JSON 解析率 | 100% | **100%** |
| schema 合法率 | 100% | **100%** |
| 动作一致率 | 69.6% | **94.6%** |
| 平均延迟 | 4.1 s | **3.0 s** |
| 平均输出字符 | 514.8 | **245.4** |
| 覆盖动作数 | 5 / 6 | **6 / 6** |

真实 9 条决策卡（训练集内，仅作复现性检查）：

| 指标 | 基座 27B | SFT 27B |
|---|---:|---:|
| 动作一致率 | 66.7% | **88.9%** |
| 平均延迟 | 3.8 s | **2.7 s** |

基座最大的问题是保守：56 条里 31 次 `do_nothing`，从不选 `hpo`，`adapt`
只出现 1 次。SFT 后 6 个动作全覆盖（`hpo` 9、`retrain_head` 11、
`recalibrate_inputs` 9、`adapt` 6、`rollback` 6、`do_nothing` 15），
且输出更短、更快。

最终权重：

```
trio://run_7yqgoj5be2n0/sampler_weights/planner-27b-sft-v1
```

接回真实 B 闭环（4 线程、`stream_mode=bearing_holdout`）：前 11 个周期
`planner=qwen`，消耗 20,966 tokens 后按预算自动回落策略；整轮
**170.7 s、峰值 84.5 MB**，与纯策略版 123.3 s / 80.9 MB 同一量级。
最终 RMSE 706.4 与策略版相同，原因是实验预算在第 3 个周期后已经用尽，
决策层再聪明也无法改变「预算耗尽 → do_nothing」的硬约束。

平衡样本里 `hpo` 与 `adapt` 刻意用不同的预算状态区分：`hpo` 的
`sweep_exhausted=false`，`adapt` 的 `sweep_exhausted=true`，且后者的诊断卡
明确写着「配置仍然合适」。否则同一张 concept 卡会有两个互相冲突的标签。

## 5. 接回 Harness

把 `harness/config/harness.json` 的 `llm.model_id` 改成 SFT 权重的
`trio://` 路径（或平台为后训练模型分配的模型 ID），其余代码不变：

```json
"llm": {
  "model_id": "trio://run_.../sampler_weights/planner-27b-sft-v1"
}
```

如果评测环境走平台自己的网关，按 `docs/平台提交表单填写指南.md` 的字段填
模型类型、权重路径、训练记录链接和 API Key；不要把 Key 写进仓库、镜像或
本文档。

## 6. 验证状态

- Harness 侧改动：`pytest tests -q` → **245 passed, 17 skipped**（2026-11-21）。
- 27B 真实卡片调用：**4.8 s / 1767 tokens / attempts=1 / thinking_off=true**，合法 JSON。
- 4 线程合成 RUL 训练：43.49 s → **31.22 s**，最终 loss 均为 0.48219。
- Docker 4 核 / 8GB / 无网络容器：研究镜像 291 MB，B 闭环 16 周期、
  训练 24.74 s、预算墙钟约 28 s、RMSE 706.435、无 OOM；提交镜像 276 MB，
  `describe` 合约正确，离线宿主驱动的一轮 C 方向 `train` 也跑通
  （52.8686 → 52.2253，adopted=yes，容器内 16.8 s）。
  详见 [docs/INDUSTRIAL_CPU_CHECK.md](../docs/INDUSTRIAL_CPU_CHECK.md)。

## 7. 已知边界

- 合成样本由规则策略和人工模板打标签，不是线上测量结果；9 条真实决策才是
  金标准。训练报告会分开统计 `real` / `policy` / `balanced`。
- `forward_backward` 会累积梯度，API 没有 `zero_grad`；因此校验损失只在
  最后一个 epoch 结束后算一次，训练中途不跑验证。
- 4 线程 BLAS 让工业模型训练快约 28%，但会改变浮点归约顺序。需要逐位复现
  历史产物时设 `BASELINE_CPU_THREADS=1`。
