# 最小可用 Harness（CPU-C 涡扇发动机 RUL）

这个目录实现赛题要求的**最小完整闭环**：面对按顺序到达的批次数据，系统在没有人工干预的情况下完成

1. **漂移检测** → 2. **诊断归因** → 3. **实验优化** → 4. **评估** → 5. **上线 / 回滚**，

并在读取下一批数据之前，先落盘本批次"用哪个模型"的决策。它不修改官方 CPU 基线包，只调用其公开训练入口并独立复算验证指标。

## 1. 五步闭环与代码映射

| 赛题要求 | 本实现 | 位置 |
| --- | --- | --- |
| 漂移检测 | 性能漂移、相对静态基线漂移、输入分布偏移、目标分布偏移 | `drift.py` |
| 诊断归因 | `concept_drift` / `covariate_shift` / `model_staleness` / `champion_regression` / `exploration` / `monitor_only` | `policy.diagnose` |
| 实验优化 | 受预算约束的续训候选（学习率倍率、epochs），从当前 champion 检查点继续 | `policy.plan_experiments` + `baseline.train_candidate` |
| 评估 | 独立加载候选检查点，在批次验证集和固定 guard 集上复算 RMSE/MAE/NASA 分数 | `baseline.regression_metrics` |
| 上线 / 回滚 | 三重门控 + 回滚到上一 champion 或官方静态基线 | `policy.gate_candidate` / `policy.rollback_decision` |

## 2. 关键设计

- **官方静态基线是硬地板**。`checkpoint.npz` 被复制到 workdir 并绑定 SHA-256；每个批次都在同一验证集和 guard 集上重新评估它。默认门控要求候选在两个集合上都不差于静态基线（`static_tolerance=0`）。
- **champion 与 previous champion 都是不可变副本**。每次 promote/rollback 都把检查点复制到 `models/champions/generation-XXXX/`，状态里记录摘要；候选目录后续被修改不会影响已上线模型。
- **固定 guard 集防止批次过拟合**。启动时从官方 `public/validation.jsonl` 确定性抽样一次，之后所有门控对比都使用同一份 guard；摘要变化会直接拒绝运行。
- **replay 防止遗忘**。候选训练数据 = 批次 `train.jsonl` + 从官方 `public/train.jsonl` 确定性抽样的 replay 行。replay 行数可配置；默认 3000。
- **训练/验证隔离是显式的**。合并训练集时会剔除批次验证集 ID 和 guard ID，并在决策中记录剔除数量；不会静默用评估行训练。
- **决策先落盘**。`run_batch` 在写完全部内容并更新 `state.json` 后才返回，主循环才读取下一批。
- **幂等与可恢复**。已处理批次默认跳过；`--force` 重跑。每个实验结束后保存状态，崩溃后可从工作目录继续。
- **失败也产出决策**。批次损坏、训练失败或超时会记录 `hold` 决策并继续下一批，不会让整个数据流停摆。
- **资源受控**。候选数量、epochs、单批时间预算、单实验超时均来自配置；默认单线程 CPU、无 GPU、无网络。

## 3. 运行契约

### 输入流

```text
stream/
  batch-01/
    train.jsonl          # 可选；缺失时使用 replay 样本（replay_rows=0 时用官方 public/train.jsonl）
    validation.jsonl     # 必需
    batch.json           # 可选，原样写入决策用于追溯
  batch-02/
    ...
```

批次按目录名排序消费，也可用 `--batch DIR` 按给定顺序显式传入。每行格式与官方包一致：`id`、`inputs`、`answer`，`meta` 可选且不会进入模型。

### 输出目录

```text
workdir/
  state.json                    # champion 血缘、计数器、批次账本
  journal.jsonl                 # append-only 审计日志（每次决策、实验、门控）
  decisions/<batch>.json        # 每批完整决策
  data/guard.jsonl              # 固定 guard 样本
  data/replay.jsonl             # 确定性 replay 样本
  data/train-<batch>.jsonl      # 本批实际训练集（含来源与剔除计数）
  models/official-baseline/     # 官方静态基线副本
  models/champions/generation-*/# 已上线检查点副本
  models/<batch>-cXX-.../       # 各候选训练输出（training.json/history/checkpoint）
```

## 4. 快速开始

依赖与官方基线一致（Python 3.10–3.12，NumPy 1.26.4）：

```sh
python -m venv .venv
# Linux/macOS
.venv/bin/pip install -r cpu-c-guide-20261110-v1/baselines/requirements.txt
# Windows: .venv\Scripts\pip install -r cpu-c-guide-20261110-v1\baselines\requirements.txt
```

生成 4 个批次的**合成演示流**（传感器标定漂移 + 目标偏移，仅用于演示闭环）：

```sh
python tools/make_demo_stream.py --output demo-stream
```

运行：

```sh
python -m harness.cli run \
  --stream demo-stream \
  --workdir runs/demo \
  --config harness/configs/demo.json
```

查看状态：

```sh
python -m harness.cli status --workdir runs/demo
```

演示流会依次触发：稳定批次的探索性提升、传感器漂移批次的自适应提升、概念漂移批次因 guard 退化被门控拒绝并回滚、以及后续批次的再次提升。演示配置放宽了门控阈值，只用于展示；正式策略请使用 `harness/configs/default.json`。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--stream DIR` | 自动发现 `batch-*/validation.jsonl`，按名字排序 |
| `--batch DIR` | 显式顺序；可重复，与 `--stream` 互斥 |
| `--dry-run` | 只做漂移分析、诊断和实验规划，不训练、不推进 champion/计数器（新 workdir 仍会初始化一次） |
| `--max-batches N` | 本次最多处理 N 批 |
| `--force` | 重新处理 state 中已有的批次 |
| `--swanlab off\|offline\|cloud\|required` | 覆盖配置中的 SwanLab 模式 |
| `--json` | 输出机器可读运行摘要 |

## 5. 策略与门控

### 漂移判定

设 champion 本批 RMSE 为 `r`，其参考 RMSE 为 `r_ref`，官方静态基线本批 RMSE 为 `r_static`：

- 性能漂移：`(r - r_ref) / r_ref`，警告 5%，严重 15%
- 相对静态：`(r - r_static) / r_static`，警告 3%，严重 10%
- 输入漂移：特征均值相对检查点标准化统计量的最大 `|z|` 与超阈值特征比例
- 目标漂移：批次目标均值偏移（以训练目标标准差计）

全部阈值在 `harness/configs/default.json` 的 `drift` 段，必须满足 `critical >= warn`，启动时校验。

### 实验规划

- 严重漂移：`max_candidates`（默认 3）
- 警告：默认最多 2 个；若尝试配额未满，则最多补足到 `min_attempts` 个
- 稳定但尝试数不足 `min_attempts`（默认 3）：补足尝试
- 稳定且配额已满：默认不训练；每 `exploration_every` 批做一次低成本探索
- 概念漂移优先较大学习率；协变量漂移优先较小学习率；所有候选都从当前 champion 续训，禁止重新初始化

### 上线门控（全部满足才 promote）

1. 候选本批 RMSE 相对 champion 的改善 ≥ `min_improvement`（默认 1%）
2. 候选本批 RMSE ≤ 静态基线 × `(1 + static_tolerance)`（默认 0）
3. 候选 guard RMSE ≤ champion guard × `(1 + guard_max_degradation)`（默认 1%）
4. 候选 guard RMSE ≤ 静态基线 guard × `(1 + static_tolerance)`
5. 指标有限、检查点可加载、特征 schema 一致

多个候选通过时，按本批 RMSE、guard RMSE、训练耗时的顺序取最优。

### 回滚

当本批没有候选通过门控，且当前 champion 相对静态基线在批次或 guard 上的劣化超过 `rollback_margin`（默认 5%）：

- 若上一 champion 在本批上至少好于当前 champion `min_improvement`，且批次与 guard 都不差于静态基线 → 回滚到上一 champion；
- 否则 → 回滚到官方静态基线（保证回滚后仍满足静态地板）。

champion 本身是静态基线时不再回滚。

## 6. 决策文件

`decisions/<batch>.json` 的核心字段：

```json
{
  "format": "harness.decision.v1",
  "workdir": "/runs/demo",
  "batch": {"id": "batch-02", "index": 2, "validation_sha256": "sha256:..."},
  "decision": "promote",
  "selected_model": {"stage_id": "...", "generation": 2, "checkpoint": "models/champions/...", "checkpoint_absolute": "/runs/demo/models/champions/.../checkpoint.npz", "sha256": "sha256:..."},
  "champion_before": {"stage_id": "...", "metrics": {"batch": {"rmse": 102.4}}},
  "reason": "候选 ... 通过门控 ...",
  "drift": {"severity": "critical", "summary": "...", "findings": [...]},
  "hypotheses": [{"code": "covariate_shift", "action": "adapt_weights"}],
  "experiments_planned": [{"epochs": 4, "learning_rate": 0.001, "hypothesis": "covariate_shift"}],
  "experiments": [{"stage_id": "...", "metrics": {"batch": {...}, "guard": {...}}, "gate": {...}}],
  "metrics": {"champion_before": {...}, "static": {...}},
  "budget": {"time_budget_seconds": 180, "elapsed_seconds": 3.2},
  "counters": {"attempts": 4, "promotions": 2, "rollbacks": 0}
}
```

`journal.jsonl` 是同一事实的追加式审计版本，包含 `batch_started`、`drift`、`diagnosis`、`training_data`、`experiment_started`、`experiment_finished`、`gate`、`decision`、`batch_error` 等事件。

## 7. SwanLab

```sh
python -m pip install swanlab
python -m harness.cli run ... --swanlab offline
```

- 记录对象是**工业模型**的批次指标、漂移信号、候选门控与决策，不是驱动 Agent 的大模型训练。
- `offline` 适合离线评测镜像；`cloud` 需要网络；`required` 在包缺失或初始化失败时直接报错。默认 `off`，SwanLab 永远不会成为运行硬依赖。
- 每批记录 champion/static/候选的 batch/guard RMSE、漂移幅度、尝试/升级/回滚计数。

## 8. 安全与资源边界

- 只读 `public/train.jsonl`、`public/validation.jsonl`、批次目录和显式命令行路径；不扫描隐藏目录，不访问网络。
- 官方基线代码与检查点只读；候选输出写入新的 workdir 目录，基线训练器自身仍会拒绝覆盖旧轮次。
- 检查点、guard、replay、训练数据均校验摘要；NPZ 由官方加载器以 `allow_pickle=False` 读取。
- 默认 `BASELINE_CPU_THREADS=1`，可通过环境变量设为 1–8；不调用 GPU。
- 时间预算覆盖本批实验阶段（候选训练与独立评估）；漂移分析为轻量 NumPy 计算，不单独计时。

## 9. 测试

```sh
python -m unittest discover -s harness/tests -v
```

测试覆盖：配置校验、漂移判定、实验规划、门控与回滚、训练集合并/剔除、状态持久化，以及端到端场景（正常流、dry-run、`--force` 索引）。端到端测试会真实调用官方 `baselines.train`，在普通 CPU 上约需 1 分钟。

官方基线测试：

```sh
cd cpu-c-guide-20261110-v1
python -m unittest discover -s baselines/tests -v
```

## 10. 当前边界与扩展点

- 本最小构建只支持 **C 赛道回归/RMSE**；A/B 分类需要扩展 `baseline.regression_metrics`、门控指标与漂移目标统计。
- 批次目录在 CLI 启动时发现；若平台逐批释放数据，可在新批次出现后用同一 workdir 再次运行，已完成批次会被幂等跳过。
- 当前策略是确定性规则，便于复现和审计；接入 PyTRiO 上的 Qwen 作为"策略 Agent"时，可让模型输出 `Hypothesis` / `ExperimentSpec`，但门控、独立评估和静态基线地板必须保留在规则层，避免模型自报自批。
- 公开数据包不含隐藏测试集；平台隐藏评测仍是最终权威。本 Harness 的本地 guard 与批次验证只是上线决策的代理指标。
- 镜像打包时应包含 `harness/`、`tools/`、官方基线包与本文件；`runs/`、`demo-stream/` 属于运行产物，不应进入镜像。
