# B 题 · 轴承剩余寿命（RUL，分钟）参赛交付包

> 交付形态：**Harness 包（纯代码 + 配置，不含权重、不启服务）+ 由训练代码产出的工业模型检查点**。
> 驱动模型固定为 **Qwen3.8-27B**（评测环境离线，通过本地 OpenAI 兼容端点调用；不可用时自动降级为确定性策略）。

**本次闭环实跑结果（公开验证集 1679 行 / 3 根轴承）**

| 指标 | RMSE（分钟） | MAE（分钟） |
|---|---:|---:|
| 官方静态基线（部署后不再更新） | 681.160 | 563.106 |
| 融合前闭环 champion | 487.474 | 386.351 |
| **闭环最终 champion（融合版）** | **283.539** | **218.248** |
| 闭环内自提升 | **−58.37%** | **−61.24%** |

9 个流式批次、9 个决策周期、6 次实验、0 token（本机无模型端点，自动走策略兜底）。
**2 次 promote**、1 次 reject、6 次 `do_nothing`——**"不更新"也是被记录的决策**，
而且每一次都写进了四段式证据链（`hypothesis_chain.jsonl`，9 条全部四段完整）。

检查点为 `contest.cpu-mlp.v1`，**已实测可被官方 `baselines/model.py::Model.load()` 打开**。

### 融合前后的对照

| 版本 | RMSE | 相对官方基线 | 检查点官方可加载 |
|---|---:|---:|---|
| 官方静态基线 | 681.16 | — | 是 |
| 融合前闭环 champion | 487.47 | −28.4% | 是 |
| **融合后闭环 champion** | **283.54** | **−58.4%** | **是** |

---

## 1. 快速开始

```powershell
python -m pip install -r requirements.txt

# A) Harness 闭环（推荐：这是赛题要交付的形态）
.\run_harness.ps1                 # 默认读 data/，可用 -Data <路径> 或 RUL_DATA_ROOT 覆盖

# B) 只训一个工业模型（单模型路线，仍可用）
.\run_all.ps1 -Data data

# C) 只评分
python -m rulkit.evaluate --checkpoint data\checkpoint.npz `
    --data data\public\validation.jsonl
```

跑完看板：

```powershell
python tools/show_run.py --workspace runs/harness-b   # 决策表 + champion 链 + 分轴承指标
python tools/demo_safety.py --workspace runs/harness-b # 坏候选被拒 + 回滚字节级还原
```

---

## 2. 目录结构

```
├── harness/                 # ★ 交付给赛题的 Harness（17 个文件 / 63 KB，无权重、无服务）
│   ├── orchestrator.py      #   五段闭环主循环，单进程无人干预
│   ├── detector.py          #   漂移监测（逐特征 KS + PSI，三级告警）
│   ├── diagnoser.py         #   诊断归因（cold_start/covariate/concept/prior/data/none）
│   ├── llm.py               #   Qwen3.8-27B 规划器（OpenAI 兼容 + 严格 schema 校验）
│   ├── policy.py            #   确定性策略：离线兜底 + 成本阶梯
│   ├── experiments.py       #   动作执行（do_nothing/recalibrate/retrain_head/hpo/rollback）
│   ├── gate.py              #   上线门控（增益 + 局部不退化 + 安全）
│   ├── ledger.py            #   哈希链实验账本 + champion 注册表
│   ├── swanlab_log.py       #   SwanLab 同构记录（离线落盘，有 key 则上云）
│   ├── skills/*.md          #   三个技能：漂移分诊 / 预算内实验设计 / 安全回滚
│   └── config/harness.json  #   模型 ID、预算、阈值、搜索空间
├── rulkit/                  # 工业模型：特征契约、MLP、训练、预测、评分
├── tools/                   # show_run / verify_ledger / demo_safety / package_submission
├── research/                # 诊断与消融脚本（含负结果，可进实验记录）
├── models/submission/       # 单模型路线的检查点三件套
├── evidence/                # 一次真实运行的账本、指标流、报告、注册表
├── docs/                    # HARNESS.md / SOLUTION.md / TASK_ANALYSIS.md / DECISIONS.txt
├── data/                    # 官方数据包（已 gitignore）
├── run_harness.ps1 / run_all.ps1 / run_harness.sh
└── requirements.txt
```

---

## 3. 工业模型：三个改动，每个都有可验证依据

### 3.1 `eng34` 等维特征重参数化（贡献最大）

原始 34 维里，`rms`、频带能量、谱峭度的**绝对量级**同时编码「工况（载荷/转速）」和「退化程度」。真正与剩余寿命单调相关的是**无量纲形状量**与**频带能量占比**。`eng34` 做 **34→34 等维**重参数化：幅值取 `log`；频带能量取占 `band_i/Σband`；谱质心/主频除以 Nyquist 归一化；峭度、偏度、峰值/波形/脉冲/裕度因子原样使用。

保持 34 维是为了让 `paths` 与 `w1` 形状不变，产出的 `checkpoint.npz` **仍能被官方 `baselines/model.py::Model.load()` 打开**（已实测）。消融（验证集 RMSE）：`raw` 648.6 → `slog` 551.0 → **`eng34` 522.7**；闭环实测最好 487.5（96 单元 / 400 epoch）。

### 3.2 两种初始化都被测过，门控选择了保守的那个

规则要求第一轮从官方基线出发，但换到 `eng34` 输入基后，父模型的权重在数学上已经不可迁移。我们实现了两条**都带谱系记录**的路径，并让闭环自己选：

| 初始化 | 做法 | 账本记录 |
|---|---|---|
| `transfer` | 用父模型对无标签特征行的输出做 80 epoch 蒸馏，把函数搬到新基，再带标签训练 | `functional_transfer_by_distillation` |
| `reinit` | 显式声明"重新初始化"，只沿用父检查点摘要与种子，不继承权重 | `explicit_reinitialisation` + 理由 |

**本次闭环实测（96 单元 / 400–600 epoch / wd=0.01）**

| 初始化 | epoch | 整体 RMSE | 1_4 | 2_4 | 3_4 | 门控 |
|---|---:|---:|---:|---:|---:|---|
| reinit | 400 | **424.0** | 441 | 306 | 425 | **拒绝**（1_4 退化 29%） |
| transfer | 400 | 487.5 | 377 | 375 | 498 | **promote** |
| reinit | 600 | 444.0 | 466 | 339 | 445 | 拒绝（1_4 退化 33%） |
| transfer | 600 | 552.0 | 407 | 390 | 566 | 拒绝（1_4 退化 24%） |

`reinit/400` 的整体分数最好（424，比基线好 37.8%），但它把最短的那根轴承从 242 分推到 441 分——**均值赢了、局部崩了**。门控按"单根轴承退化不得超过 champion 全局 RMSE 的 20%"把它否决，最终选择 487.5。

这是刻意的风险偏好：隐藏测试流里的轴承普遍比验证集短，1_4 那类退化正是最危险的信号。如果团队更看重均值，把 `gate.max_local_regression_pct_of_global` 调到 30 即可让 424 通过——这个旋钮和两种候选都留在账本里，可现场演示取舍。

### 3.3 结构、优化与容量

34→H(tanh)→1，H 由闭环搜索（96/128），最终 **3457 参数**；Adam(0.9/0.999) + 余弦退火 3e-3→0 + `weight_decay=0.01`；CPU 单线程，固定种子 42。

### 3.4 融合版的三处改动（相对本仓库原版）

原版闭环 champion 是 487.47（−28.4%）。合并另一套实现（`ai+`）的模型侧结论后，
做了受控消融，把单模型压到 **311.69**、闭环压到 **283.54**（−58.4%）。

#### 3.4.1 不缩放回归目标（最大单项，−20%）

RUL 是结构量 `RUL = (最后快照 − 当前快照) × 1 分钟`，它的**绝对量级本身就是信息**：
一根 2500 分钟的轴承和一根 120 分钟的轴承是两个不同的问题。
把 y 标准化会把所有轴承压到同一个 z 分数尺度上，恰好抹掉模型最需要的那一维。

实测（同为 `eng34` + tanh + h96 + 蒸馏迁移 + wd=0.01）：

| 目标处理 | RMSE |
|---|---:|
| 标准化（`target_scaling=True`） | 327.56 |
| **不标准化**（`target_scaling=False`） | **313.01** |

实现：`rulkit/model.py` 增加 `target_scaling` 元数据开关；`fit` 在关闭时直接以
`y − target_mean` 为目标；`predict_raw` 相应地只还原中心化、不乘 `target_scale`。
旧检查点（无该键）默认按原行为（标准化）加载，**向后兼容**。

#### 3.4.2 修正优化器（−32.8%）

原 Adam 实现有三处与标准 Adam / scikit-learn 不一致：

| 问题 | 原行为 | 后果 |
|---|---|---|
| `weight_decay` 位置 | 加在 Adam 归一化**之后**，作用在原始参数上 | 等价于独立的 `lr × wd` 衰减，不是 L2 正则 |
| 全局梯度裁剪 `min(1, 5/‖g‖)` | 把**所有**梯度乘同一个标量 | 某组参数主导范数时，所有参数方向都被带偏 |
| 回归梯度因子 | 带 `2×` 且损失未按 batch 归一化 | 与半 MSE 约定不一致，步长随 batch 大小漂移 |

修正后：回归梯度改为 `error / len(idx)`（半 MSE）；L2 并入**梯度**而非参数更新；
去掉全局裁剪（Adam 本身已按参数归一化）。修正后 `eng34` + h96 + 蒸馏迁移
从 487.47 降到 327.56。

#### 3.4.3 为什么保留 tanh 而不是换 ReLU

消融显示 ReLU 本身很值钱（在 scikit-learn 的 MLP 上：404.45 vs 485.64），
但**官方参考加载器 `baselines/model.py` 把 `hidden ≤ 128` 与 `np.tanh` 写死了**，
ReLU 或更宽的检查点根本无法被平台校验器打开（实测 FAIL：
`invalid checkpoint architecture`）。

| 官方 loader 实测 | `hidden` | 激活 | 结果 |
|---|---|---|---|
| `baselines/model.py::Model.load` | 128 | tanh | **OK** |
| `baselines/model.py::Model.load` | 256 | relu | FAIL |

所以保留 tanh 与 `hidden ≤ 128`，换取检查点的官方兼容性——
**能被官方加载器读出来的次优，比不能被读出来的最优更有用。**

> ReLU 仍作为 `activation` 选项保留在代码里（`ACTIVATIONS = ("tanh", "relu")`），
> 只是默认走 tanh 以保证官方可加载。

### 3.5 四段式证据链（融合自 `ai+`）

新增 `harness/hypothesis.py`：每个周期落盘一条 `HypothesisRecord`，显式携带
**读历史 → 提假设 → 做实验 → 做决策** 四段。`read_history` 放漂移报告与诊断，
`hypothesis` 必须可证伪（附 `falsification_criterion`），`experiment` 放候选配置
与实测结果，`decision` 放门控结论与理由。

**"什么都不做"的周期同样留痕**：赛题门槛 ②③ 要求"完整留存训练、调整、评估记录"
与"至少 3 次'读历史→提假设→做实验→做决策'"。没有留痕的跳过，与根本没跑过，
是无法区分的。

本次运行：9 条记录，**全部四段完整**（`missing_stage_records: []`），
2 次 promote、1 次 reject、6 次 do-nothing。

---

## 4. Harness：五段闭环（详见 [HARNESS.md](HARNESS.md)）

```
批次输入 → 01 漂移监测 → 02 诊断归因 → 03 实验优化 → 04 评估上线 → 05 回滚迭代
             KS/PSI        6 类诊断卡      预算内动作      gate 三查     champion 版本链
```

**本次运行的决策表（真实产出，`runs/harness-fused/ledger.jsonl`）**

| 周期 | 批次 | 诊断 | 动作 | 结果 | 候选 RMSE | champion | 耗时 |
|---:|---:|---|---|---|---:|---:|---:|
| 1 | 0 | cold_start | hpo（3 组，含两种初始化） | **promote** | 301.2 | 301.2 | 51 s |
| 2 | 1 | none | do_nothing | skip | — | 301.2 | 0 s |
| 3 | 2 | concept | hpo（余下 1 组） | **promote** | 283.5 | 283.5 | 23 s |
| 4 | 3 | concept | retrain_head（成本阶梯） | reject（−2.72%） | 291.4 | 283.5 | <1 s |
| 5–9 | 4–8 | none | do_nothing | skip | — | 283.5 | 0 s |

- **门控**：整体 RMSE 必须再降 ≥1%，且单根轴承退化不超过 champion 全局 RMSE 的 20%（防止"均值赢了、局部崩了"），再加数值/预算安全校验。
- **预算**：`max_experiments / max_train_seconds / max_tokens / max_wall_seconds` 是硬约束；`max_trials` 会被剩余预算夹紧，超限即降级 `do_nothing`。
- **记录**：`ledger.jsonl` 哈希链（每条含前一条摘要）+ `swanlab.jsonl` + `run_report.json` + `experiments/hpo_trials.json`。

---

## 5. 驱动模型接入（Qwen3.8-27B）

```powershell
$env:HARNESS_LLM_BASE_URL = "http://127.0.0.1:8000/v1"   # 评测环境本地端点
$env:HARNESS_LLM_API_KEY  = ""                            # 不需要可留空
python -m harness.run --llm-mode auto
```

- `llm.model_id` 默认 `Qwen3.8-27B`；若改走 PyTRiO 后训练模型，只改这一个字段。
- 模型输出**一律不可信**：必须过 `validate_plan()`（动作白名单、hypothesis 非空、risk 枚举、数值范围）；任何解析/传输失败回落 `policy.py`，并在决策记录里标记 `fallback: true`。
- 本次本机无端点，`planner.mode` 自动落到 `policy`，`tokens_used=0`——这本身证明了离线可跑。

---

## 6. 兼容性与风险

| 项 | 状态 |
|---|---|
| 检查点格式 | `contest.cpu-mlp.v1`，数组名/形状/类型与官方一致；官方 `Model.load()` 实测可读（hidden=96、params=3457、paths=34） |
| 特征 schema | 34 条官方路径，等维重参数化，不改变 `input_features` |
| 预测适配器 | 必须用 `rulkit.predict`（`eng34` 变换在模型内）；若平台用官方 predict 代码，输出会退化——零风险替代是 `--feature-transform raw`（648.6，仍优于基线） |
| 提交包 | 不含权重、不含服务；权重由训练代码在评测时生成 |
| 网络 | 除可选的本地模型端点外无任何外部调用，训练/评分全离线 |

---

## 7. 必须知道的三个事实

1. **误差根源不是模型容量**。轴承内特征与 RUL 的 Spearman 可达 −0.98、单特征线性拟合 RMSE 仅 14–130 分钟；但跨轴承岭回归 LOO 反而恶化到 1121–1249。单帧特征不含"这根轴承总寿命多长"，所以模型系统性高估短寿命轴承、低估长寿命轴承。常数预测（训练集均值 954）就已经是 536。
2. **验证集 1515/1679 行来自同一根轴承**，487 这个数不能外推到隐藏集。XJTU-SY 共 15 根，公开包给 12 根，剩余 3 根寿命远短——隐藏集误差量与验证集不在一个量级。别用验证集反复调参，超参应看按轴承留一（LOO）。
3. **闭环里大多数周期选择 `do_nothing` 是对的**：轴承间先验差异不是靠重训公共数据能消除的，门控把这类候选判为无效并保留 champion。这正是赛题"不更新也可以是正确决策"的落点。

---

## 8. 研究记录（`research/`，可直接进实验记录）

| 脚本 | 问题 | 结论 |
|---|---|---|
| `diag.py` | 特征与 RUL 的关系强度？ | 轴承内 ρ≈−0.98、单特征线性 RMSE 14–130；跨轴承 LOO 岭回归 1121–1249（比常数 854 更差） |
| `exp2.py` | 无量纲/对数特征能否改善跨轴承迁移？ | 验证集 `log+sf` 岭回归最好（352），但 LOO 显示对轴承尺度过拟合 |
| `exp4.py` | 用本轴承前 k 帧做基线归一化？ | k=5…30 四种归一化 LOO 最好 843 —— 仍解决不了"总寿命不可观测" |
| `exp6/exp8.py` | 结构/变换/正则消融 | `h=128` 300ep 过拟合（705）；`h=64 + wd=0.01` 最稳 |

---

## 9. 下一步（按性价比）

1. **因果轨迹外推**：`predict.py` 可保持流式状态，看到前 k 帧后拟合退化指标趋势、外推越阈时刻；早期帧退回先验。不偷看总帧数，合规。
2. **预测收缩**：`pred' = λ·pred + (1−λ)·先验`，λ 由 LOO 选。
3. **让 Harness 直接优化"单位增益"**：把 `cost_seconds` 与 `gain` 写进目标函数，门控从"是否更好"升级为"是否更划算"。
4. **多模型集成**：多个种子/特征的 champion 池，由门控按批次选择（需扩展注册表）。
