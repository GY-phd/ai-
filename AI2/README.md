# 智训引擎 · 轴承剩余寿命自进化 Harness

**Industrial model self-evolution harness** — drift monitoring → diagnosis → budgeted experimentation → gated promotion → rollback, closing the loop on a streaming bearing RUL task (XJTU-SY, minutes) with a 3.5K-parameter CPU model and Qwen3.8-27B as the decision layer.

<p>
  <code>Python 3.10+</code> · <code>NumPy only</code> · <code>CPU</code> · <code>no weights in the harness package</code>
</p>

> 赛题 9「智训引擎」——工业制造场景下的模型训练与超参数优化（B 题：滚动轴承剩余寿命预测）
> 第二届北京市大学生"人工智能+"创新大赛 · 出题：软通计算机有限公司、华为技术有限公司

---

## 成绩

| 指标（公开验证集 1679 行 / 3 根轴承） | RMSE（分钟） | MAE（分钟） |
|---|---:|---:|
| 官方静态基线（部署后不再更新） | 681.160 | 563.106 |
| 常数预测（训练集均值） | 536.271 | — |
| **本系统闭环最终 champion** | **487.474** | **386.351** |

9 个流式批次 · 9 个决策周期 · 6 次实验（全部计入预算）· 457 秒训练 · 3,457 参数。
决策构成：1 次 promote、2 次 reject、1 次"候选未过安全筛"、5 次 `do_nothing`。

---

## 融合版成绩（本次合并）

在上述系统之上合并了另一套实现（`ai+`）的模型侧结论，重新做了受控消融，
**单模型**在同样的官方留出轴承上达到：

| 方案 | 特征变换 | 激活 | 目标缩放 | 宽度 | RMSE | 相对基线 |
|---|---|---|---|---:|---:|---:|
| 官方静态基线 | — | — | — | — | 681.16 | — |
| 融合前闭环 champion | eng34 | tanh | 是 | 96 | 487.47 | −28.4% |
| 消融：raw + tanh + 缩放 | raw | tanh | 是 | 96 | 623.35 | −8.5% |
| 消融：raw + tanh + 不缩放 | raw | tanh | 否 | 96 | 522.64 | −23.2% |
| 消融：slog + tanh + 不缩放 | slog | tanh | 否 | 96 | 506.40 | −25.6% |
| 消融：eng34 + tanh + 缩放 | eng34 | tanh | 是 | 96 | 327.56 | −51.9% |
| **消融：eng34 + tanh + 不缩放** | eng34 | tanh | 否 | **96** | **301.18** | **−55.8%** |
| 消融：eng34 + relu + 不缩放 | eng34 | relu | 否 | 128 | 639.68 | −6.1% |

**闭环跑完后的最终 champion：RMSE 283.54 / MAE 218.25（−58.4%）**，
2 次 promote、1 次 reject、6 次 do-nothing，3,457 参数。

消融可复现：`python -m research.exp_fusion`（每行一个配置，负结果也保留）。

三处改动，都可单独复现：

1. **不缩放回归目标**（最大单项）。RUL 是结构量
   `RUL = (最后快照 − 当前快照) × 1 分钟`，它的**绝对量级本身就是信息**
   ——一根 2500 分钟的轴承和一根 120 分钟的轴承是两个不同的问题。
   把 y 标准化会把所有轴承压到同一个 z 分数尺度上，恰好抹掉模型最需要的那一维。
   实测：`scaling=True → 327.56`，`scaling=False → 301.18`（同宽度 96）。
2. **修正优化器**（原仓库的 Adam 实现有三处与 scikit-learn / 标准 Adam 不一致）：
   - `weight_decay` 被加在 Adam 归一化**之后**，作用在原始参数上，
     等价于一个 `lr × wd` 的独立衰减，而不是 L2 正则；
   - 全局梯度裁剪 `min(1, 5/‖g‖)` 把**所有**梯度乘同一个标量，
     某一组参数一旦主导范数，所有参数的方向都会被带偏；
   - 回归梯度带 `2×` 因子而损失没有按 batch 归一化，与半 MSE 约定不一致。
   修正后同一配置 RMSE 从 487.47 降到 327.56（−32.8%）。
3. **教师蒸馏迁移初始化**。父模型的函数被搬到新输入基上，再带标签训练。
   两条路径（`transfer` / `reinit`）都留有谱系记录，由门控按风险偏好选。

### 为什么保留 tanh 而不是换 ReLU

消融显示 ReLU 本身很值钱（在 scikit-learn 的 MLP 上：404.45 vs 485.64），
但**官方参考加载器 `baselines/model.py` 把 `hidden ≤ 128` 与 `np.tanh` 写死了**，
ReLU 或更宽的检查点根本无法被平台校验器打开。
所以这里保留 tanh 与 `hidden ≤ 128`，换取检查点的官方兼容性——
**能被官方加载器读出来的次优，比不能被读出来的最优更有用。**

| 官方 loader 实测 | `hidden` | 激活 | 结果 |
|---|---|---|---|
| `baselines/model.py::Model.load` | 128 | tanh | **OK** |
| `baselines/model.py::Model.load` | 256 | relu | FAIL（`invalid checkpoint architecture`） |

---

## 为什么值得一看

1. **系统否决了自己找到的最好模型。** 扫参里 `reinit/400` 整体 424.0（比基线好 37.8%），但把最短那根轴承从 242 推到 441（+29% 全局 RMSE）。门控按"单根退化 ≤ 全局 RMSE 的 20%"拒绝它，最终选 487.5——均值赢了、局部崩了，在这道题里是陷阱。
2. **"不更新"是有效决策。** 9 个周期里 5 次 `do_nothing`。轴承间寿命先验差异不是重训公共数据能消除的，门控把这类候选判为无效并保留 champion。
3. **离线也能闭环。** 驱动模型是 Qwen3.8-27B，但端点不可用时自动降级为确定性策略（同动作空间、同 schema），`tokens=0` 仍跑通全流程，并把 `fallback:true` 写进决策记录。
4. **四段式证据链。** `harness/hypothesis.py` 把"读历史 → 提假设 → 做实验 → 做决策"四段显式落盘到 `hypothesis_chain.jsonl`，**包括"什么都不做"的周期**——没有留痕的跳过，和根本没跑过，是无法区分的。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 把官方数据包放好（见 data/README.md）
#    data/public/train.jsonl  data/public/validation.jsonl  data/checkpoint.npz

# 2) 跑一遍闭环（约 8 分钟：6 组训练 + 9 个批次）
python -m harness.run --workspace runs/harness-b --batch-size 200 --max-experiments 6

# 3) 看结果
python tools/show_run.py --workspace runs/harness-b
python tools/verify_ledger.py --ledger runs/harness-b/ledger.jsonl
```

Windows 上用 `.\run_harness.ps1`；只想离线跑策略不调模型：`--llm-mode policy`。

<details>
<summary>接 Qwen3.8-27B（评测环境提供的本地端点）</summary>

```bash
export HARNESS_LLM_BASE_URL=http://127.0.0.1:8000/v1   # OpenAI 兼容
export HARNESS_LLM_API_KEY=...                          # 可留空
python -m harness.run --llm-mode auto
```

模型 ID 固定在 `harness/config/harness.json` 的 `llm.model_id`（默认 `Qwen3.8-27B`）。
换 PyTRiO 后训练的 4B/27B 检查点只改这一个字段。返回的 JSON 一律过 `validate_plan()`
（动作白名单 / hypothesis / risk / 数值范围），任何解析或传输失败都回落到 `harness/policy.py`。
</details>

---

## 系统镜像（赛题交付物）

赛题要求提交「模型配置 + Harness 的系统镜像」。仓库根目录的 `Dockerfile` 就是它：

```bash
docker build -t zhixun-engine-b .

# 评测机的运行形态：离线、无人工干预、时限受控
docker run --rm --network none \
  -v /path/to/official-data:/data:ro \
  -v /path/to/output:/out \
  zhixun-engine-b
```

镜像的四个关键性质：

| | |
|---|---|
| **零参数运行** | 入口是 `python -m harness.run`，配置全走环境变量（`RUL_DATA_ROOT` / `HARNESS_WORKSPACE` / `HARNESS_BATCH_SIZE` / `HARNESS_MAX_EXPERIMENTS` / `HARNESS_MAX_WALL_SECONDS` / `HARNESS_LLM_MODE`），不需要传任何参数 |
| **离线可用** | 只依赖 `numpy`；**刻意不装 swanlab**——评测环境离线，本地已按 SwanLab 形状写指标流，云端客户端只会给一次不该失败的运行增加失败模式 |
| **可离线闭环** | 大模型端点不可达时自动降级为确定性策略，`tokens=0` 仍跑完全流程；**已实测**（socket 全阻断时 `available()` 返回 False 并抛 `LLMUnavailable`，由策略接管） |
| **不可能带权重** | 构建时执行 `tools/verify_repo.py`，权重文件会直接让构建失败 |

以非 root 用户（uid 10001）运行；`/out` 是产物目录，评测机在其上取
`run_report.json` / `ledger.jsonl` / `hypothesis_chain.jsonl` / `swanlab.jsonl`。

---

## 架构

```mermaid
flowchart LR
    A[批次输入] --> B[01 漂移监测<br/>KS + PSI]
    B --> C[02 诊断归因<br/>6 类诊断卡]
    C --> D[03 实验优化<br/>预算内动作]
    D --> E[04 评估上线<br/>gate 三查]
    E --> F[05 回滚迭代<br/>champion 版本链]
    F -.-> A
    E -.->|候选无效| G[保留 champion<br/>= 自动回滚]
    G -.-> A
```

| 阶段 | 模块 | 关键设计 |
|---|---|---|
| 监测 | `harness/detector.py` | 逐特征两样本 KS（对重尾谱特征比 PSI 稳）+ PSI 报告；`ok/warn/alarm` 三级 |
| 归因 | `harness/diagnoser.py` | `cold_start / covariate / concept / prior / data / none`，决定要不要花标注 |
| 实验 | `harness/experiments.py` | `do_nothing < rollback < recalibrate_inputs < retrain_head < hpo`；扫参去重、候选先过安全筛 |
| 门控 | `harness/gate.py` | 整体增益 ≥1% **且** 单根退化 ≤ 全局 RMSE 的 20% **且** 数值/预算安全 |
| 记账 | `harness/ledger.py` | 哈希链账本（每条含前一条摘要）+ champion 注册表 |
| 记录 | `harness/swanlab_log.py` | 离线写 SwanLab 同构 JSONL；有 key 时镜像上云 |

---

## 仓库结构

```
├── harness/            决策层与闭环（18 个文件，无权重）
│   ├── orchestrator.py   主循环：监测→诊断→实验→门控→回滚
│   ├── detector.py       KS/PSI 漂移监测
│   ├── diagnoser.py      诊断归因
│   ├── llm.py            Qwen3.8-27B 规划器 + schema 校验
│   ├── policy.py         确定性策略（离线兜底）
│   ├── experiments.py    动作执行与候选筛选
│   ├── gate.py           promote / reject 判定
│   ├── ledger.py         哈希链账本
│   ├── hypothesis.py     四段式证据链（读历史→提假设→做实验→做决策）
│   ├── swanlab_log.py    实验记录
│   └── skills/*.md       三个技能：漂移分诊 / 预算内实验 / 安全回滚
├── rulkit/             工业模型：特征契约、NumPy MLP、训练、预测、评分
├── tools/              show_run / verify_ledger / verify_repo / demo_safety /
│                      package_submission / build_github_repo
├── tests/              55 个单元测试（不依赖官方数据包，CI 与本地都能跑）
├── research/           诊断与消融脚本（含负结果，exp_fusion 即融合消融）
├── evidence/           本次运行的账本、四段式证据链、指标流、注册表
├── docs/               HARNESS.md（设计）/ SOLUTION.md（方法）/ TASK_ANALYSIS.md / DECISIONS.txt
├── .github/workflows/  CI：编译 + 单测 + 账本校验 + 零权重自检 + 决策表渲染
└── data/               放官方数据包（已 gitignore）
```

> `models/submission/`（含权重）**不随仓库分发**。检查点由训练代码在评测时生成，
> `evidence/registry/champion.json` 里记录了可复现它的命令。

---

## 复现一次闭环

```bash
python -m harness.run --workspace runs/harness-b --batch-size 200 --max-experiments 6 --config harness/config/harness.json
```

产物：`ledger.jsonl`（决策账本）、`hypothesis_chain.jsonl`（四段式证据链）、
`swanlab.jsonl`（指标流）、`run_report.json`（总报告）、
`registry/`（champion 版本链）、`experiments/`（每个候选的检查点）。

仓库内 `evidence/` 是**一次真实运行**的留档，可直接对照：

```bash
python tools/show_run.py --workspace evidence     # 决策表
python tools/verify_ledger.py --ledger evidence/ledger.jsonl
# OK  9 records, 9 decisions, 2 promotions, chain intact
python tools/verify_repo.py --dir .                # 零权重自检
```

---

## 质量保障

```bash
python -m unittest discover -s tests -t . -v     # 55 tests, ~1 s
python -m compileall -q harness rulkit tools tests
```

覆盖的是**会咬人的地方**，不是凑数：

| 测试 | 保护什么 |
|---|---|
| `test_features` | 34 维官方 schema 顺序、禁止 label 派生字段、`eng34` 频带占比求和为 1 |
| `test_model` | 检查点保存/加载逐位一致、`expand` 保函数（新单元出边为 0）、参数上限 |
| `test_gate` | 整体增益 ≥1% 且单根退化 ≤ 全局 RMSE 20%；局部退化按全局尺度衡量 |
| `test_ledger` | 哈希链可验证、篡改与断链都能被抓到、champion 指针可解析 |
| `test_policy` | 6 类诊断到动作的映射、预算耗尽降级、被拒后沿成本阶梯下探 |
| `test_config` | `RUL_DATA_ROOT` 重定位、workspace 绝对化、模型 ID 固定为 Qwen3.8-27B |
| `test_fusion` | 四段式证据链的缺失阶段检测、**旧格式检查点仍可加载**、激活/目标缩放开关 |

CI（`.github/workflows/ci.yml`）在 **ubuntu + windows × Python 3.10/3.12** 上跑同一套，
并且每次 push 都会重新校验仓库内 `evidence/ledger.jsonl` 的哈希链、重新渲染决策表——
**证据文件被改动，CI 就会红**。



1. **瓶颈不是模型容量，是"轴承总寿命不可观测"。** 同一轴承内特征与 RUL 的 Spearman 可达 −0.98、单特征线性拟合 RMSE 仅 14–130 分钟；跨轴承岭回归 LOO 反而劣化到 1121–1249。所以绝对 RMSE 有先验下限，常数预测（954）就已经是 536。
2. **等维特征重参数化是最有效的单点改进。** 把绝对量级（耦合工况）换成无量纲比值 + 对数，验证集 648.6 → 522.7，且保持 34 维官方 schema，参考加载器仍可读。
3. **两种初始化要实测，不能假设。** 教师蒸馏迁移（487.5）与从零初始化（424.0）差 12%，但后者在短轴承上退化；最终由门控按风险偏好选择。
4. **门控阈值就是风险偏好。** `gate.max_local_regression_pct_of_global` 从 20 改到 30，424 的候选立刻通过——两种结果都在账本里，可现场演示取舍。

---

## 已知限制

- 验证集高度集中（1515/1679 行来自同一根轴承），487 这个数不能外推到隐藏测试流；隐藏集是 3 根更短的轴承，误差量级会不同。
- `eng34` 特征变换在模型内，推理须用 `rulkit.predict`；若只允许官方 predict 代码，改 `--feature-transform raw`（648.6，仍优于基线 4.8%）。
- Qwen 端点的 A/B（模型规划 vs 策略表）尚未联调，是本项目下一步。

---

## License

代码 MIT（见 [LICENSE](LICENSE)）。文档与图表同许可；官方赛题材料不随仓库分发，见 [data/README.md](data/README.md)。
