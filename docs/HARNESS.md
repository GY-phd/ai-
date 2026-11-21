# Harness：工业模型自进化闭环（B 题 · 轴承 RUL）

> 交付形态：**只有代码和配置**。不含权重、不启动任何服务、不做任何外部网络调用（除可选的本地模型推理端点）。
> 工业模型本身由 `rulkit.train` 在运行时训练产生；决策层是被指定的驱动模型 **Qwen3.8-27B**。

```
harness/
├── orchestrator.py   # 主循环：监测 → 诊断 → 实验 → 门控 → 回滚，单进程无人干预
├── detector.py       # 漂移监测（逐特征 KS + PSI，ok/warn/alarm 三级）
├── diagnoser.py      # 诊断归因（cold_start / covariate / concept / prior / data / none）
├── llm.py            # Qwen3.8-27B 规划器（OpenAI 兼容协议 + 严格 schema 校验）
├── policy.py         # 确定性策略：离线/模型不可用时的同构兜底
├── experiments.py    # 动作执行：do_nothing / recalibrate / retrain_head / hpo / rollback
├── gate.py           # 上线门控：整体增益 + 局部不退化 + 预算与安全校验
├── ledger.py         # 实验账本（哈希链）+ champion 注册表
├── swanlab_log.py    # 实验记录：离线写 SwanLab 同构 JSONL，联网时镜像上云
├── skills/*.md       # 三个技能文档：漂移分诊 / 预算内实验设计 / 安全与回滚
├── config/harness.json
└── run.py            # python -m harness.run
```

## 1. 与赛题要求逐条对应

| 赛题要求 | 本实现 |
|---|---|
| 无人干预完成全部阶段 | `orchestrator.Harness.run()` 一个进程跑完 9 个批次，无任何 `input()` 或人工确认 |
| 至少 3 次"读历史→提假设→做实验→做决策" | 9 条 `decision` 记录，含 hypothesis / action / cost / gate；账本可逐条回放 |
| 综合性能不低于官方静态基线 | 门控强制：候选 RMSE 必须比 champion 低 ≥1%，否则 `reject` 并保留原模型；基线 681.16 是启动值，最终 487.47 |
| 完整留存训练/调整/评估记录 | `ledger.jsonl`（哈希链）+ `swanlab.jsonl` + `experiments/hpo_trials.json` + `run_report.json` |
| Harness + 模型 ID + SwanLab 记录 | 模型 ID 在 `config/harness.json` 的 `llm.model_id`；SwanLab 记录离线落盘，配置密钥后自动上云 |
| 提交包不含权重、不跑服务 | 包内只有 `.py/.md/.json`（143 KB），权重由训练代码在评测时生成；唯一网络调用是可选的本地模型端点 |

### 本次实跑（`runs/harness-b/`）

| 周期 | 诊断 | 动作 | 结果 | 候选 | champion |
|---:|---|---|---|---:|---:|
| 1 | cold_start | hpo ×3（reinit/transfer 两种初始化） | promote | 487.47 | 487.47 |
| 2 | concept | hpo ×1（余下配置 552.0） | skip：未过局部退化筛 | 552.0 | 487.47 |
| 3 | concept | retrain_head | reject（−1.14%） | 493.0 | 487.47 |
| 4–8 | none | do_nothing | skip | — | 487.47 |
| 9 | concept | retrain_head | reject（−1.14%） | 493.0 | 487.47 |

扫参里整体最好的候选是 `reinit/400 = 424.0`（比基线好 37.8%），但它把最短的 1_4 从 242 分推到 441 分，被"单根退化 ≤ 全局 RMSE 20%"的安全筛否决；系统最终选了更保守的 487.47。**这个否决是产物的一部分**：账本里同时保留了 424 的完整指标，把 `gate.max_local_regression_pct_of_global` 调到 30 就能让它通过。

安全演示 `tools/demo_safety.py`：把输出层放大 3 倍的坏候选（RMSE 781.75）送进门控 → `reject`；随后 `rollback` 还原 champion，**字节级一致**（sha256 相同）。

## 2. 五段闭环

```
批次输入 ──▶ 01 漂移监测 ──▶ 02 诊断归因 ──▶ 03 实验优化 ──▶ 04 评估上线 ──▶ 05 回滚迭代
              (KS/PSI)        (6 类诊断卡)      (预算内动作)      (gate 三查)      (champion 版本链)
                   │                                                        │
                   └──────────────── 预算耗尽自动降级为 do_nothing ◀─────────┘
```

- **监测**：逐特征两样本 KS（对重尾谱特征比 PSI 稳），`share_shifted` 与 `ks_max` 双阈值分级。
- **归因**：`cold_start`（从未适配过）、`covariate`（输入移、关系没变）、`concept`（误差涨得比漂移快）、`prior`（整体偏置但序关系还在）、`data`（退化到常数通道）、`none`。
- **实验**：按成本从低到高 `do_nothing` < `rollback` < `recalibrate_inputs` < `retrain_head` < `hpo`；`hpo` 在公开训练集上训练候选并逐轮记录。
- **门控**：① 整体 RMSE 改善 ≥1%；② 单根轴承退化不超过 champion 全局 RMSE 的 20%（防止"均值赢了、局部崩了"）；③ 数值有限、预算未超。
- **回滚**：champion 注册表保存上一版指针，`rollback` 字节级还原；被拒绝的候选同样入账。

## 3. 驱动模型（Qwen3.8-27B）

```powershell
$env:HARNESS_LLM_BASE_URL = "http://127.0.0.1:8000/v1"   # 评测环境提供的本地端点
$env:HARNESS_LLM_API_KEY  = "..."                        # 不需要可留空
python -m harness.run --llm-mode auto
```

- 协议：OpenAI 兼容 `/chat/completions`，`model` 取自配置（默认 `Qwen3.8-27B`）。
- 若走 PyTRiO 后训练路线，只需把 `llm.model_id` 改成平台上注册的模型 ID（例如 4B 后训练版），代码不变。
- **输出不可信**：返回的 JSON 必须过 `validate_plan()`——动作白名单、hypothesis 非空、risk 枚举、数值范围；任何解析/传输失败都回落到 `policy.py`，并把 `fallback: true` 与原因写进决策记录。
- `--llm-mode policy` 可完全离线跑通全流程，用于开发和复现。

### 3.1 模型身份解析：全仓库唯一真源

模型 ID 有两套拼写——平台下拉框用斜杠，赛题文本用点号：

| 用途 | 写法 | 说明 |
|---|---|---|
| 平台下拉框（PyTRiO） | `Qwen/Qwen3.8-27B` | 发请求时用这个 |
| 赛题文本 / 本项目配置 | `Qwen3.8-27B` | 配置里写这个 |
| 后训练派生（路线二） | `qwen3.8-27b-ft-20261115` | 以基座合规 |

**唯一真源**：`harness/llm.py`。全仓库任何模块要判断"这是哪个模型 / 是否合规"都**必须**调它的入口，不得自带第二份规则。

| 入口 | 回答的问题 | 语义 |
|---|---|---|
| `resolve_model_id(model_id)` | **用哪种拼写发请求？** | 点号短名 → 斜杠平台名；已是斜杠的原样返回；未知 ID 原样透传；空输入返回空串 |
| `is_allowed_model(model_id, allowed, derived)` | **这个模型允许用吗？** | 三层判定：① 显式派生豁免 → ② 归一化后**包含**允许基座 → ③ 否则拒绝 |
| `normalize_model_id(model_id)` | 两个拼写是不是同一个模型 | 忽略大小写与所有分隔符，并**去掉厂商前缀**（`Qwen/` 是命名空间不是模型名） |

**派生模型为何放行**：路线二的模型名被平台改写过（带 `-ft-<日期>` 后缀），字面相等校验会把**合规**的模型误判为违规。所以判定用「归一化后包含基座」——`qwen3.8-27b-ft-20261115` 归一化后包含 `qwen3827b`，自动放行，无需维护后缀长名单。

**两条规则不合并**：`resolve_model_id` 只管拼写，`is_allowed_model` 只管合规。**把一个非法模型名改写成合法拼写会掩盖违规**——改写后的请求携带的是运维从未选择的模型。所以判定函数永不改写 ID。

`evo/`（早期对照实现）也调同一入口，不带兜底副本：**兜底副本仍是第二份副本**，两份会漂移，而没人读的那份才是最终决定合规与否的那份。

## 4. 预算记账

`budget` 段是硬约束，不是提示：

| 项 | 含义 | 超限行为 |
|---|---|---|
| `max_experiments` | 允许的实验次数 | 直接降级 `do_nothing` |
| `max_train_seconds` | 累计训练秒数 | 同上 |
| `max_tokens` | 决策模型 token | 不再调用模型，走策略表 |
| `max_wall_seconds` | 整轮墙钟 | 同上 |

每次决策都记录 `cost_seconds`（实测）与 token 用量，报告里直接给出"单位增益消耗"。

## 5. 运行

```powershell
# 完整闭环（B 题验证流，9 个批次）
python -m harness.run --workspace runs/harness-b --batch-size 200

# 只跑策略、不调用模型（离线复现）
python -m harness.run --llm-mode policy --max-experiments 3

# 指定配置
python -m harness.run --config harness/config/harness.json
```

产物：

```
runs/harness-b/
├── ledger.jsonl          # 哈希链决策账本
├── swanlab.jsonl         # 指标流（SwanLab 同构）
├── registry/             # champion.json + champions.jsonl（版本链）
├── experiments/          # 每个候选的 checkpoint + hpo_trials.json
└── run_report.json       # 基线/最终指标、增益、预算、逐轮决策
```

## 6. 设计取舍（评审可能追问）

1. **为什么保留确定性策略？** 评测环境可能拿不到模型端点；没有兜底的 Harness 在端点抖动时会直接失效。策略表与模型共享同一动作空间和同一 schema，切换对下游无感。
2. **为什么门控用公开验证集？** 数据包明确写"public/validation.jsonl 用于衡量改动"。封闭评测时同一路径换成"已到达标签的回放缓冲"，代码不变。
3. **为什么 `do_nothing` 出现这么多次？** 本任务的漂移主要是轴承间寿命先验差异，重训公共数据并不改变隐藏流的先验；门控会把这些候选判为无效。**"不更新也可以是正确决策"正是指标之一**。
4. **为什么不把 checkpoint 打进包？** 规则要求 Harness 包不含权重；权重由训练代码在评测环境生成，`training.json` 记录父检查点摘要，谱系仍然可核验。
