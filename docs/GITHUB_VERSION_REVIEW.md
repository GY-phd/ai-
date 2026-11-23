# GitHub 库内版本审查报告（队友提交版）

> 审查对象：`origin/main` @ `4054a20`（"作品文档：C 方向最终提交说明与镜像哈希"）
> 审查时间：2026-11-23
> 审查方式：独立 worktree 检出 `origin/main`，跑通全量测试，逐文件对照本地基线 `26bbeed`，核对账本与文档数字
> 本地基线：`26bbeed`（"本地跨版本 CI 矩阵 + 系统 Python 升级"）
> **本报告不改动任何代码，也未向 GitHub 推送任何内容。**

**本文档结构**：
- **§1–§7** —— 完整审查报告（内部视角，含完整推导与验证过程）
- **§8** —— **给队友的回复（可直接转发）**，措辞面向作者、每条附证据位置，可整节复制发出

---

## 0. 结论先行

队友这一版**总体质量很高**：架构抽象干净、评估口径重构有明确理由、负结果如实记录、
全量测试 359 项全绿、账本数字与文档表述**逐项对得上**。不是"能跑就行"的水平。

但有 **3 个 P0 级问题**必须在提交前处理，其中最严重的一个
**已被官方平台实测确认**：C 方向这次镜像的隐藏测试 RMSE **56.25329**，
比官方静态基线 **52.8686** 差 **6.40%**，
**违反平台"效果不低于平台基线"的硬性要求**。

平台报告的逐轮数据还给出了更精确的诊断：

| 口径 | 平台值 | 与隐藏 56.25 的差 |
|---|---:|---:|
| 公开验证 RMSE（部署的 r5） | 51.111160 | **+10.06%** |
| 最低公开验证 RMSE（r7，未被采用） | 50.907431 | +10.51% |
| 官方静态基线 | 52.8686 | −3.38 |
| 离线切片 `late_rmse`（平台不可见） | 33.9105 | +22.34 |
| 离线 `proxy_rmse`（平台不可见） | 28.5025 | +27.75 |

**两个关键结论**：

1. **公开验证确实低估隐藏难度**（C 落差 +10.06%，而 B 隐藏反而更好 10.78）
   —— `docs/C_WEIGHTING.md` §0 的核心担忧**方向是对的**。
2. **但切片口径在平台上不可见、不可审计**，且 `proxy_rmse` 与平台指标差 27.75。
   **r7 全量最低却未被采用**，说明只看全量也不够 —— 正确解法是**并列判据**，
   而非"只看切片"或"只看全量"。

| 级别 | 数量 | 概要 |
|---|---:|---|
| P0 | 3 | C 全量指标回归被当作增益；Docker 镜像源被写死国内镜像；README 头表用不可部署配方报成绩 |
| P1 | 4 | 离线账本丢字段（guard/理由不可见）；`docs/TECHNICAL_REPORT.md` 已过期；`_bounded_prior_weights` 大 top 值 OOM；`delta()` 文档与实现不符 |
| P2 | 5 | `harness/llm.py` 缓存键含时间量；`enable_thinking` 首包可能被拒；A 方向结论未随新代码更新；命名不一致；分支管理混乱 |

---

## 1. 这一版做了什么

7 个提交，`+59526 / -156` 行，70 个文件。核心是四块：

### 1.1 Adapter 插件化（新增 `contest_baselines/adapters/`）

新增 `base.py` / `profile_a.py` / `profile_b.py` / `profile_c.py` / `registry.py` / `hypotheses.py`，
把此前散落在 `strategies.py`、`train.py` 里的 `if profile == "A"` 分支收敛成插件。

契约设计得很到位：

- `task` / `classes` / `group_of` / `label_of` / `supports` / `hypotheses` / `epoch_scale` /
  `extra_metrics` / `test_horizon` / `weight_horizon` / `selection_view` / `selection_guard` /
  `resize_hidden`
- 明确写出 **"特征 schema 不得改变"** 的约束：平台预测请求只带 `{task, inputs}`，
  没有 sample id，所以任何从 id 派生的列在隐藏集上无法复算 —— 这条被反复强调，
  是正确的架构判断
- `registry.py` 对未知 key **抛异常而不是猜**，理由写得很清楚

> **评价：这块可以直接采纳。** 它把"加第四个 profile 要改四个函数"降成"加一个模块 + 注册表一行"，
> 并且没有引入任何隐式耦合。

### 1.2 评估口径重构（`RoundController` + `Adapter.selection_view`）

这是这一版**最大也最危险**的改动。

C 方向的隐藏测试协议是"每台发动机在截断点出一行"，而公开验证集是"完整序列每一行"。
全量 RMSE 因此**不是**隐藏指标的代理。作者引入了一个"协议对齐切片"：

- `Adapter.test_horizon()` → C 返回 `200.0`（依据公开 `RUL_FD00x` 标签上界 ~195）
- `Adapter.selection_view()` → 在验证集上取 `y <= 200` 的行算 RMSE，记为 `late_rmse`
- `Adapter.selection_guard()` → 允许全量 RMSE 回退 **12%**
- `compare()` 改为比较 `selection_value`；全量指标退化为被守护的旁证

设计意图是清楚的：**"全量验证越好的模型，在隐藏协议上不一定越好"**。
`docs/C_WEIGHTING.md` 用 86 条配置记录的搜索证据支撑了这个判断，不是拍脑袋。

### 1.3 训练侧逐行损失权重（`sample_weight`）

在 `rulkit/model.py` 和 `contest_baselines/model.py` 的 `fit()` 里加了可选 `sample_weight`。
新增假设 `the-tail-is-underweighted-by-weight`（`1 + unit-relative remaining life`）
和 `the-test-is-truncated`（有界均匀先验 / 训练 RUL 密度的重要性比，开方 temper，`floor=0.05`）。

> **评价：实现是干净的。** 我验证了 `weights=ones` 时与无权重路径**逐位相同**
> （参数与 loss 都 `allclose` 为 True），权重不落进 checkpoint，
> `Model.load()` 每轮新建对象所以**不会跨轮泄漏**。加权梯度与所报 loss 解析一致。

### 1.4 27B 决策层可靠性（`harness/llm.py`）

针对上一版"288 秒超时 + 空正文"的真实故障，做了系统性加固：

- 端点探测缓存（`probe_ttl_s=30`）
- 计划缓存（相同状态复用，命中记 0 token）
- 历史记录投影（`compact_history()`，只保留决策需要的字段）
- 三段降级：`enable_thinking=False` → 无参 → 放大 token 预算
- 空正文 / 非法 JSON 分别重试并记 `failures`
- `<think>...</think>` 剥离
- 配置新增 `HARNESS_LLM_MODEL` 环境变量覆盖（不改镜像即可换模型）

> **评价：这是这批改动里工程价值最高的一块。** 文档记录的真实诊断卡
> 4.8 s / 1767 tokens / 合法 JSON，对比旧实现 288 s + 空正文，是可验证的改善。

---

## 2. P0 问题（必须处理）

### P0-1 C 方向第 3 轮是一次真实回归，但被记录成了增益 🔴

> ### ⚠️ 修正（2026-11-23 平台实测后）
>
> **推理链要改，结论不变。**
>
> 我最初从离线账本推断"平台部署的是 `offline-artifact-3` = 57.42"。
> **平台实测证明这个推断不成立**：
>
> | 轮 | 平台公开验证 RMSE | 离线账本 full RMSE | 平台动作 |
> |---:|---:|---:|---|
> | 3 | **52.225263** | 57.4199 | 未训练，保留父模型 |
> | 5 | **51.111160** | 57.4199 | 采用本轮模型（最终） |
> | 7 | **50.907431** | 57.4199 | 未训练，保留父模型 |
>
> **平台上根本没有 57.42 这个产物** —— 平台 r3 是"未训练，保留父模型"。
> 平台实际部署的是 **r5 产物，公开验证 RMSE 51.111160**。
>
> **原因**：平台由真实 27B 决策层 + 真实漂移数据流驱动，
> 与离线重放的轨迹完全不同。**离线 10 轮不能代表平台 10 轮。**
>
> **但 P0-1 的结论依然成立**，而且现在有平台实测数字：
> **C 方向隐藏 RMSE = 56.25329，比官方基线 52.8686 差 6.40%**，
> **违反平台"效果不低于平台基线"的硬性要求**。
>
> 完整数据见 `docs/PLATFORM_REPORT_REVIEW.md`。

**这是本次审查最重要的发现。**

> ### 🚨 已被官方平台实测确认（2026-11-23 补充）
>
> 队友把镜像提交到官方平台跑完，**隐藏测试 RMSE = 56.25329 循环**，
> **比官方静态基线 52.8686 差 6.40%**，
> **违反平台"效果不低于平台基线"的硬性要求**。
>
> ⚠️ 但要注意：**平台部署的不是离线账本里那个 57.42 的产物**（见上方修正框）。
> 平台实际部署的是 **r5 产物，公开验证 RMSE 51.111160**。
>
> **这带出一个更关键的问题**：平台的公开验证与隐藏之间有 **+10.06%** 的落差
> （51.11116 → 56.25329），而 B 方向隐藏反而比公开好 10.78。
>
> **所以 `docs/C_WEIGHTING.md` §0 的核心担忧方向是对的** ——
> 全量公开验证**确实不是**隐藏指标的可靠代理；
> **但他的解法有问题**：`selection_view` 的切片指标在平台报告里
> **完全不可见、不可审计**，而且没有与全量指标形成任何联合判据。

**离线账本上发生了什么**（记录在案，作为"离线≠平台"的例证）：

`runs/harness-offline-c-weight/ledger.jsonl` 第 3 轮：

```
r3  the-test-is-truncated   adopted=True   parent_advanced=True
    rmse 52.225 → 57.420   （+9.95%）
    late 38.520 → 33.911   （−11.96%）
```

**离线部署产物的谱系**（`run_report.json` 的 `parent_advanced` 字段可证）：

| 轮 | 假设 | 采纳 | 父模型变为 | 离线部署产物 |
|---:|---|:-:|---:|---|
| 1 | more-data-helps | ✅ | 52.225 | offline-artifact-1 |
| 2 | the-parent-was-left-unconverged | ❌ | 52.225 | offline-artifact-1 |
| 3 | **the-test-is-truncated** | ✅ | **57.420** | **offline-artifact-3** |
| 4–10 | tail / unit / capacity / more-data ×4 | ❌ | 57.420 | offline-artifact-3 |

**r4–r10 全部被拒，父模型一直停在 57.42**。
12% 的 guard 拦不住这次回归 —— 9.95% < 12%，它是被**规则允许**采纳的。
guard 的作用只是防止"塌掉"，而 57.42 并没有塌掉，它只是系统性地更差。

> 🔑 **而平台 r3 是"未训练，保留父模型"，根本没有 57.42 这个产物。**
> 这说明 **离线重放的 10 轮 ≠ 平台上的 10 轮**，
> 真实 27B 决策层 + 真实漂移数据流下轨迹完全不同。
> 队友的 104 条配置搜索、88 条实验记录同理 —— **都不能直接当作平台结果来推断**。

#### 2.1.1 平台实测与三个预测口径的对照（决定性证据）

队友把镜像提交到官方平台，隐藏测试结果：

| 指标 | 平台实测值 |
|---|---:|
| RMSE | **56.25329 循环** |
| MAE | 39.938255 循环 |
| NASA 非对称评分 | 123 654 600 886.84648 |
| 隐藏测试样本 | 9 687 |
| 完成轮次 | 10 / 10 |
| 最终采用 | 第 5 轮候选 1 |
| CPU 用时 / 运行用时 | 7 859.56 s / 6 160.48 s |

**把这个实测值与平台**公开验证**指标对照 —— 这才是平台口径下的正确比较：**

| 口径 | 平台值 | 与隐藏 56.25 的差 | 说明 |
|---|---:|---:|---|
| **公开验证 RMSE**（平台报告列） | 51.111160 | **+5.14（+10.06%）** | 平台部署的 r5 产物 |
| 最低公开验证 RMSE（r7） | 50.907431 | +5.35（+10.51%） | 全场最低，但**未被采用** |
| 官方静态基线 | 52.8686 | −3.38 | **比隐藏成绩还好** |
| 离线 `late_rmse`（切片） | 33.9105 | +22.34 | 平台**不显示**这个口径 |
| 离线 `proxy_rmse`（截断代理） | 28.5025 | +27.75 | 平台**不显示**这个口径 |

**结论**：

1. **C 方向公开验证系统性低估隐藏难度，落差 +10.06%。**
   对照 B 方向（隐藏 609.08 **优于**公开 619.86），
   C 的落差方向相反、幅度也大得多。
   **所以 `docs/C_WEIGHTING.md` §0 的核心担忧方向是对的** ——
   全量公开验证**确实不是**隐藏指标的可靠代理。
2. **但解法有问题。** 平台报告只列**公开验证指标**，
   `selection_view` 的切片值**在平台上完全不可见、不可审计**。
   一个评委看不到的口径做选型判据，**等于没有判据**。
3. **关键线索：r7 的公开验证 50.907431 是全场最低，却"未训练，保留父模型"，
   最终采用了 r5 的 51.11116。**
   **唯一合理的解释是选型确实用了某个非全量的口径** ——
   很可能就是队友的 `selection_view`。
   也就是说**切片机制在平台上生效了**，
   但**它没有让 C 变得比基线好**。
   （平台不提供各轮次的隐藏成绩，所以无法判断 r5 到底比 r7 好还是差。）
4. **`proxy_rmse` 那一整套搜索（104 条配置）与平台指标完全对不上** ——
   差了 27.75。这个指标在平台上不可见、不可验证，
   **用它做搜索目标，等于在优化一个裁判看不到的量**。

> 🔑 **修正后的建议**：
> 不是"把 `selection_view` 降级为只记录"，而是
> **"让 `selection_view` 与全量指标形成联合判据，并且把两个口径都写进可审计的记录里"** ——
> 因为平台数据显示**只看全量也确实不够**（r7 全量最低却未必最优）。
> 正确做法是**并列条件**：全量不劣化 **且** 切片改善，两者同时满足才采纳，
> 并且 guard 恢复 5% + 绝对下限。

#### 2.1.2 平台与离线轨迹不一致（关键例证）

平台显示"**最终采用 第 5 轮候选 1**"，但离线账本里 **r5 是被拒的**
（`the-unit-is-the-confounder`，`adopted=False`）。
平台 C 的逐轮公开验证与离线账本**逐轮都不同**：

| 轮 | 平台公开验证 | 离线账本 full | 平台动作 |
|---:|---:|---:|---|
| 3 | **52.225263** | **57.4199** | 未训练，保留父模型 |
| 4 | 51.153125 | 57.4199 | 采用本轮模型 |
| 5 | 51.111160 | 57.4199 | 采用本轮模型（最终） |
| 7 | **50.907431** | 57.4199 | 未训练，保留父模型 |

**平台上根本没有 57.42 这个产物。**

原因：
- 平台有真实 27B 决策层参与，假设选择与离线策略不同
- 平台数据流是真实漂移流，与离线重放不同

这一点**本身不是 bug**，但它有一个**方法论后果**：
**离线的 10 轮不能代表平台上的 10 轮**，
所有基于离线 run 的结论（104 条配置搜索、88 条实验记录）都有这个外推风险。
**这也是为什么队友那套 C 加权体系最终在平台上没跑出来。**

#### 2.1.3 其他平台观察

| 项 | 观察 |
|---|---|
| 隐藏样本 **9 687** | C-MAPSS 官方 test 是 707 台发动机（100+259+100+248）。9 687 **远大于 707**，说明**平台隐藏集不是"每机一行"**，而是多机多周期采样。这与队友假设的"每台发动机只在截断点出一行"**不符** —— 这可能是 `late_rmse` 切片无法对齐隐藏协议的直接原因。 |
| **r9 的 NASA 异常跳变** | r7/r8 NASA = 4.79e9，**r9 突然变成 1.40e10（2.9 倍）**，而 RMSE 只从 50.907431 变到 51.245666（+0.66%）。**RMSE 几乎没变而 NASA 涨 2.9 倍**，说明 r9 出现了少量预测值大幅偏离（可能预测了负 RUL 或极端值）。值得单独查 r9 的预测分布。 |
| NASA 评分量级 | 最终隐藏 NASA = 1.2365e11。若按 `max(p,y)/10 - min(p,y)/10`，9687 样本最坏约 1.9e6，**差 5 万倍**。说明平台的 NASA **不是**这个定义，或累加了其他项。队友代码里的 `nasa_asymmetric_score` 实现与平台口径很可能不一致 —— 它是"参考指标"不参与 RMSE 评分，但**技术报告里不能拿它做结论**。 |
| 运行用时 6 160 s | 10 轮约 616 s/轮。C 是三方向里单轮最贵的。 |
| CPU 7 859 s vs 运行 6 160 s | 差值 1 699 s（约 28%）是评测端开销，正常。 |
| **轮次预算利用率** | A 训练 6 轮 / B 训练 3 轮 / C 训练 4 轮。**B 方向 10 轮里 7 轮空转**，且 r8 比 r5 劣化却仍保留 —— B 几乎已收敛，而 B 恰恰是本地改进最大（+39%）的方向。 |
| **A 方向落差最大** | 公开验证 Macro-F1 0.986738，隐藏只有 **0.87572**，掉 **0.111**。而 README 头表写 A = 0.9707（研究口径），**比隐藏实测高 0.095**。 |
| 27B 用量 | 30 轮决策共 16 032 Token（输入 14 755 / 输出 1 277），平均每轮约 535 Token。说明决策输出很简短，不是长推理链 —— 符合"CoT 会超时"的结论。 |

更麻烦的是，`docs/C_WEIGHTING.md` §0 的结论是这样写的：

> 上一轮 C 加权被门控拒绝，原因是公开验证全量 RMSE 从 51.239 变成 51.705。
> 重开后确认：**这不是加权无效，而是评估口径和目标口径不一致。**

"评估口径不一致"这个判断本身我认为是**成立的**。但由此推出的"所以改用切片判定"这一步，
**只验证了切片会变好，没有验证切片对隐藏集是好的**。整个论证链里，
`late_rmse` 与隐藏协议的相关性始终是**假设**，从未被独立数据证实过 ——
因为私有评测包是另一批数据，没有任何本地证据能验证它。

**风险清单**（按严重度）：

1. ~~隐藏包 RUL 分布若不像公开 `RUL_FD00x`（上界 195），`horizon=200` 这个界就是错的~~
   → **已被平台数据证实**：真实 RMSE 56.25 就是全量 RMSE 的量级，
   切片（33.91）和 proxy（28.50）**都远低于真实值**，说明这两个口径衡量的
   根本不是隐藏集在评的东西
2. guard 只防"塌掉"不防"系统性变差"，12% 的容差恰好放过了这次 +9.9% 的回归
3. 12% 是在**单个离线 run** 上定的，没有跨种子/跨划分的稳健性证据
4. `docs/C_WEIGHTING.md` 自己写了"静态基线前沿停在 proxy ~31.7"，
   而最终方案的 proxy 28.5 —— 也就是说**收益主要来自一个未经外部验证的口径**
5. **【新增】平台隐藏集 9 687 个样本 ≠ 707 台发动机**，
   说明"每机一行截断快照"的协议假设本身就不成立，
   `selection_view` 的切片设计从根上就是针对了一个错误的协议模型

**建议（保守优先，按顺序）：**

- **立刻（平台数据已给出 ground truth，不用再跑离线对照）**：
  **改成并列判据** —— 采纳条件 =「**全量不劣化**」**且**「切片改善」，
  而不是现在的"切片改善 + 全量允许劣化"。
  依据：平台 r7 的全量 50.907431 是全场最低却未被采用，
  **说明只看全量不够**；但切片值平台不可见，
  **单独用它判定等于无判据**。两个口径必须同时进决策、同时进记录。
- **立刻**：把 `selection_guard()` 恢复为 **0.05**，
  并加上**绝对下限**（全量劣化超过基线 2% 一律不采纳）。
- **立刻**：**把切片值写进平台可见的记录**（ledger / SwanLab / run_report），
  让"为什么 r7 没被采用、为什么 r5 被采用"**可审计**。
  现在这个决策在平台上是黑盒 —— 评委看不到，也没法复核。
- **立刻**：**把 `selection_view` 的口径定义写进技术报告**，
  说明它对应隐藏协议的什么假设、边界在哪（`horizon=200` 从哪来）。
- **中期**：**重新审视 104 条配置记录的搜索目标**。
  `proxy_rmse` 与平台指标差 27.75，且平台不可见、不可验证 ——
  用它做搜索目标等于在优化一个裁判看不到的量。
  要么改用**全量 RMSE** 重新搜，要么换与平台指标同分布的口径。
- **中期**：**查清 C 的 10 轮为什么有 6 轮空转**
  （平台 r2/r3/r6/r7/r8/r9/r10 都是"未训练，保留父模型"）。
  `the-test-is-truncated` 在平台 r3 根本没触发，
  说明**离线成立的假设在平台上不成立** —— 轮次预算没用满是直接原因。
- **中期**：**保留 `docs/C_WEIGHTING.md` §0 的核心论断**
  （平台数据证实了公开验证低估隐藏难度，落差 +10.06%），
  但**补上它缺的那一半**：这个论断只在"有平台数据"时成立，
  离线无法验证，所以**任何基于切片的调参都不能当结论**。
- **最优**：`selection_view` 保留为**证据**（它确实提供了"晚期区间表现"这个观察维度），
  但**绝不能让它单独决定采纳**。若要利用它，让它做**并列条件**
  （全量不劣化 **且** 切片改善），而不是"切片改善 + 全量允许劣化"。

> ⚠️ 我没有改动任何代码，这条建议留给你决策。
> **好消息**：平台已经给出了 ground truth，修正路径清楚，
> **不需要新的离线实验**（而且我上一轮说过"补一个离线对照 run" ——
> **那条建议现在作废**，因为离线 run 与平台轨迹根本不同，对照了也说明不了问题）。

### P0-2 Dockerfile 把 pip 源写死成国内镜像 🔴

```dockerfile
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt
```

**两个问题叠加：**

1. **默认值被改成了镜像源。** `ARG` 的默认值直接生效，官方评测机如果访问不了
   Tsinghua 源，**镜像构建直接失败**，整个提交报废 —— 而当前 `requirements.txt`
   只有 `numpy>=1.26,<3`，走官方 PyPI 完全没问题。
2. **`ARG` 声明位置在 SDK 安装之后**（第 79 行 vs 第 66 行），
   `pip install /opt/contest-harness-sdk` 不走镜像源 —— 也就是说 SDK 走官方源、
   numpy 走镜像源，**两条链路不一致**，这显然不是本意。

**建议**：把默认值改回官方源，镜像源做成**可选覆盖**：

```dockerfile
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt
```

如果确实需要国内加速，用 `--extra-index-url` 而不是 `--index-url`，
这样镜像源挂掉时仍能回落到官方源。

### P0-3 README 头表把不可部署配方报成了系统成绩 🔴

`README.md` 第 37–39 行的头表：

| 方向 | 基线 | 系统 | 增益 |
|---|---:|---:|---:|
| A | 0.7240 | **0.9707** | **+34.07%** |
| B | 681.160 | **636.671** | **+6.54%** |
| C | 52.869 | **44.493** | **+15.84%** |

这三个数字来自 `artifacts/rmse_verification.json`，其 recipe 里写着：

```json
"C": {"recipe": {"hidden": 128, "epochs": 200, "target": "cycles",
                 "extra_features": "elapsed", ...}}
```

`extra_features: "elapsed"` —— **"已运行循环数"**。这个特征在隐藏请求里拿不到，
因为平台预测请求只有 `{task, inputs}`，没有 cycle、没有 unit。

**`docs/TECHNICAL_REPORT.md` 自己是清楚的**，它明确写了：

> 研究 final 的 44.49 用了 `elapsed`；隐藏请求没有 cycle/unit，不可复现。
> 不把不可部署的 389/44.49 当作提交成绩。

而且它给的**可部署**数字是 A 0.7277 / B 639.33 / C 50.57。

**代码里也有明确的自我认定**，`research/tracks/optimize_abc.py` 第 12–18 行：

> the **deployable** score (a model that only sees ``inputs`` at
> prediction time; no unit id, no elapsed count).
> The current research champions for B (two-stage) and C (``elapsed``
> feature) are **not deployable**: the platform's prediction request carries
> only ``{task, inputs}``.

也就是说，**这套代码自己写明了 README 头表报的那三个数字不可部署**，
而 README 把它们放在了"智训引擎"一栏，和官方基线并排做对比。

**所以同一个仓库里，两个文档对同一件事给了两套矛盾的口径**：
技术报告是诚实的，README 头表不是。而**评委先看 README**。

**建议**：README 头表改成可部署口径（研究口径另起一张表，明确标注
"研究侧，不可作为提交成绩"），并在头表下面加一行指向
`docs/TECHNICAL_REPORT.md` 的口径说明。

> 顺带：本地基线分支上你昨天修的 B 侧 `fitted_lifetime` 把官方切分做到 414.73，
> **优于**这版 README 里的 636.67。合并时 B 的数字需要重算，
> 否则会把本地已有的更好结果覆盖成更差的。

---

## 3. P1 问题（提交前应处理）

### P1-1 离线账本丢字段，门控证据不可见 🔴

`tools/run_harness_offline.py` 第 278–292 行的账本写出器是**硬编码字段列表**：

```python
for record in report["datasets"][key]["rounds"]:
    ledger.append({
        "round": ..., "hypothesis": ..., "adopted": ...,
        "validation_before": ..., "validation_after": ...,
        "selected_artifact_id": ..., "model_call_ids": ...,
    })
```

而 `RoundController.record()` 已经把 `selection_metric` / `selection_before` /
`selection_after` / `selection_guard` / `reason` 写进了 entry。
**离线账本把这些字段全丢了。**

后果很直接：我审查时无法从账本看出第 3 轮"为什么被采纳"、
回退了多少、守住了 12% 里的多少 —— 必须读代码才能推断。
**这正好解释了 P0-1 为什么会被漏掉**：证据链在离线这一步就断了。

**建议**：把写出器改成透传 entry 的全部字段（或至少补上
`reason` + `selection_*` + `selection_guard`），并补一条断言：
账本里的 `validation_after` 必须与 `selection_after` 同时存在。

### P1-2 `docs/TECHNICAL_REPORT.md` 已过期且与代码矛盾 🟡

报告日期 2026-11-21，C 加权那批提交是 2026-11-22。报告里明确写：

> 提交侧曾实现 `the-tail-is-underweighted-by-weight` 并由 controller 透传 `sample_weight`；
> 单测覆盖方向性和可拟合性。10 轮实测 r4 被门控拒绝后，**已从 B/C adapter 撤回**；
> `sample_weight` 能力保留在研究/未来。

**但代码里 `THE_TEST_IS_TRUNCATED` 已经加进 C adapter 并在 r3 被采纳了。**
报告 §8 还写着"若 SDK 允许，应在轮次记录里同时报告最后周期代理指标" ——
而这一版**已经改了判定逻辑本身**，不只是"同时报告"。

**建议**：报告要么更新到新状态（并把 P0-1 的结论写进去），
要么明确标注"本文描述 11-21 状态，C 加权口径重构见 `docs/C_WEIGHTING.md` §6"。
**不要让评委在两份互相矛盾的文档之间猜。**

### P1-3 `_bounded_prior_weights` 在大 RUL 值上会 OOM 崩溃 🟡

```python
top = int(max(np.ceil(float(values.max())), np.ceil(float(horizon))))
grid = np.arange(0, top + 1, dtype=float)
kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / float(bandwidth)) ** 2)
```

构造了一个 **(top+1) × (top+1) 的稠密矩阵**。实测：

| top | kernel 显存 |
|---:|---:|
| 362（C-MAPSS 实际最大 RUL） | 1.0 MB ✅ |
| 5 000 | 191 MB ⚠️ |
| 20 000 | 3.0 GB ❌ |
| 10^9 | 直接 `numpy._core._exceptions._ArrayMemoryError` ❌ |

当前 C-MAPSS 的最大 RUL 约 362，所以**现在不会炸**。但：
- 私有评测包是**另一批同领域数据**，RUL 量级如果不同就可能踩到
- 这是 `MemoryError` 而不是优雅降级，会**直接把整轮打断**

**建议**：把稠密核换成**一维卷积**（`np.convolve` 或 `scipy.ndimage.gaussian_filter1d`），
或者对 `top` 夹一个上限（`min(top, 4000)`）并在夹断时记录一条 changed 说明。
核的对角带宽只有 8，稀疏化或卷积都不改变数学结果。

### P1-4 `metrics.delta()` 的文档与实现不符 🟢

```python
def delta(before, after):
    """How much the primary metric moved. ..."""
    return float(after.selection_value) - float(before.selection_value)
```

docstring 说 "primary metric"，实现返回的是 **selection** 差值。
引入 `selection_view` 之后两者可以不同，这个 docstring 现在是错的。

影响很小（`delta` 在 `contest_baselines` 里没有调用点，`tools/run_harness_offline.py`
有自己的 `_delta`），但**这是个公开函数**，评委会读。

**建议**：把 docstring 改成 "the selection metric" 并说明与 primary 的关系。

---

## 4. P2 问题（可延后）

| # | 问题 | 说明 |
|---|---|---|
| P2-1 | `harness/llm.py` 缓存键含 `budget` | `budget` 里含 `tokens_left` 等随时间变化的量，等价于状态但缓存命中率会低于预期。可考虑只取决策相关子集。 |
| P2-2 | `enable_thinking=False` 可能被端点拒 | 首包就带这个参数，若 PyTRiO 对未知字段返 400，`_post` 抛的异常会走 `LLMUnavailable` → 落到第二段"无参重试"。**降级链是对的**，但第一段的 400 会被计入 `failures`。建议第一段失败时**不**计 failure（它是探测性参数，不是真故障）。 |
| P2-3 | A 方向结论未随新代码更新 | `docs/ABC_OPTIMIZATION.md` 说 A 的下一步是 `sample_l2`，`artifacts/tracks/a_transforms.json` 已产出，但 A 的 adapter / strategies 没有对应改动，A 的 10 轮 run 里 `more-data-helps` 连续 7 次被拒。建议明确 A 的下一步落点。 |
| P2-4 | 命名不一致 | `the-tail-is-underweight` 与 `the-tail-is-underweighted` 两个 key 并存（`strategies.py` 里用 `in ("the-tail-is-underweight", "the-tail-is-underweighted")` 兼容），容易误用。建议统一成一个。 |
| P2-5 | 远端分支管理混乱 | `origin/` 上有 `GY-ai+-2`(71 文件)、`copilot/merge-three-system-directions`(238)、`feat/minimal-harness`(240)、`feature/qwen-sft`(372)、`feature/task-a-engine`(97) 等分支，`main` 是唯一完整的一条。建议明确"以 main 为准，其他分支归档或删除"，避免后续合并踩坑。 |

---

## 5. 核对结果（可信的部分）

为避免只报问题不给出正面结论，以下项目我**实际验证过**：

### 5.1 测试全绿

```
359 passed, 1 skipped, 1 warning, 117 subtests passed in 145.86s
```

（Python 3.12 venv，在独立检出的 `origin/main` 上跑的）

### 5.2 文档数字与账本逐项对得上 ✅

`docs/TECHNICAL_REPORT.md` §5 的 10 轮数字，我用 `runs/harness-offline/ledger.jsonl`
逐轮核对，**全部一致**：

| 方向 | 报告声称 | 账本实测 | 相对基线 | 采纳/拒绝 |
|---|---:|---:|---:|---:|
| A | 0.7140 → 0.7192 | 0.714 → 0.7192 ✅ | +0.73% | 1 / 9 ✅ |
| B | 681.16 → 456.42 | 681.1598 → 456.4174 ✅ | +33.0% | 9 / 1 ✅ |
| C | 52.87 → 50.68 | 52.8686 → 50.6834 ✅ | +4.13% | 6 / 10 ✅ |

最差候选、恢复轮次也都能对上。**这份报告的数字是诚实的。**

### 5.3 加权实现正确性 ✅

- `weights = ones` 时参数与 loss 与无权重路径**逐位相同**
- 加权梯度与所报 loss 解析一致（回归 `2we/mass`、分类 `(w/mass)(p−onehot)`）
- `sample_weight` **不写入 checkpoint**，`Model.load()` 每轮新建对象 → **不跨轮泄漏**
- `contest_baselines/model.py` 与 `rulkit/model.py` 各自内部前后一致
  （前者用 `2e/n` 惯例，后者用 `e/n` 的 sklearn 惯例，两者互不影响）

### 5.4 无权重入库 ✅

`git ls-files` 确认**没有任何 `.pkl` / `.pt` / `.safetensors` 被跟踪**。
本地 `evo/artifacts/*.pkl` 是未跟踪的残留文件，符合你之前"不要放权重"的要求。
仓库体积健康（`size-pack: 23.37 MiB`，`artifacts/` 共 2.3 MB）。

### 5.5 模型 ID 命名统一 ✅

| 位置 | 值 |
|---|---|
| `harness/config/harness.json` | `Qwen3.8-27B`（短名） |
| `harness/llm.py` `MODEL_ALIASES` | `qwen3.8-27b` → `Qwen/Qwen3.8-27B` |
| `train/sft_planner.py` | `Qwen/Qwen3.8-27B`（平台名） |

`resolve_model_id()` 大小写不敏感、支持 `qwen3_8_27b` 等写法、支持
`<base>-ft-<date>` 派生模型。**P2「统一 27B/4B 模型 ID 命名」这项本地待办，
在远端这一版已经解决了。**

### 5.6 仓库卫生 ✅

- `MANIFEST.json` 已随新文件重新生成（403 文件 / 6.44 MB）
- `.gitignore` 正确排除了 `train/data/`、`train/reports/`
- `Dockerfile.research` 里**移除了** `COPY runs/ ./runs/`（研究镜像不该带运行产物）✅
- 27B **不在本地推理**，镜像不带权重 ✅

---

## 6. 建议的处理顺序

**第 1 步（先做，影响提交结论）**

1. 补一个**关闭 selection_view 的 C 对照 run**，与现有 run 对比最终模型 → 定 P0-1
2. 改 Dockerfile 的 `PIP_INDEX_URL` 默认值 → 定 P0-2
3. 改 README 头表为可部署口径 → 定 P0-3

**第 2 步（证据链完整性）**

4. 离线账本写出器补字段 → P1-1
5. 更新 `docs/TECHNICAL_REPORT.md`（或标注版本边界）→ P1-2
6. `_bounded_prior_weights` 换卷积 / 夹 `top` 上限 → P1-3

**第 3 步（合并本地工作）**

7. 把本地 `fitted_lifetime` 的 B 侧修复（官方切分 414.73，优于远端 636.67）
   合并到 adapter 体系，重算 README / `rmse_verification` 的 B 数字
8. 修 `tests/test_the_closed_loop_beat_its_own_initial_champion` 与 README 数字审计
   （本地遗留的两项，本轮未动）
9. 清理 `origin/` 的分支，明确 main 为唯一基线 → P2-5

**第 4 步（收尾）**

10. 跑 `python tools/refresh_manifest.py` + `pytest tests -q` + `python -m tools.verify_repo`
11. 重建镜像做一次 4 核 / 8 GB 实跑（`docs/INDUSTRIAL_CPU_CHECK.md` 里的流程）
12. 确认无误后再决定是否推送（当前按你的要求**未推送**）

---

## 7. 附：审查方法与可复现命令

```powershell
# 独立检出远端版本，不动本地工作区
git fetch origin
git worktree add --detach "d:\VSCode project\ai+_audit_origin" origin/main

# 全量测试（隔离 tmp，避免 Windows 目录权限噪声）
cd "d:\VSCode project\ai+_audit_origin"
& "d:\VSCode project\ai+\.venv\Scripts\python.exe" -m pytest tests -q `
  --no-header -p no:cacheprovider --basetemp="$PWD\.pytest_tmp_audit"

# 逐轮核对账本
git show origin/main:runs/harness-offline/ledger.jsonl
git show origin/main:runs/harness-offline-c-weight/ledger.jsonl

# 清理
git worktree remove --force "d:\VSCode project\ai+_audit_origin"
```

> 说明：审查期间在 worktree 内产生的 `.pytest_tmp_audit/` 临时目录已删除，
> 未对 `origin/main` 做任何写操作。

---

## 8. 给队友的回复（可直接转发）

> 本节是 §2–§4 结论的**对外版本**：措辞面向作者、每条附可复现的证据位置，
> 并先讲清已验证通过的部分。可以整节复制发出。

### 8.0 开场

先说结论：**这一版整体质量很高**，Adapter 插件化的抽象很干净，
27B 决策层的可靠性加固（探测缓存 + 计划缓存 + 历史压缩 + 三段降级）是实打实的工程改进，
`docs/TECHNICAL_REPORT.md` 里的 10 轮数字我用账本逐轮核对过，**全部对得上**，负结果也如实记录了。

但有 **3 个 P0** 建议在提交前处理，其中第 1 条最关键 ——
**它已经在官方平台上被实测确认**：这次镜像的隐藏测试 RMSE 56.25329，
比官方静态基线 52.8686 差 6.40%，**不满足平台"效果不低于平台基线"的硬性要求**。

我把平台报告的逐轮数据抄下来了，**最值得注意的是这段**：

> **r7 的公开验证 RMSE 50.907431 是全场最低，却被判"未训练，保留父模型"；
> 最终采用了 r5 的 51.11116。**

这一行同时说明了两件事 —— **只看全量不够**（r7 全量最低却未必最优），
**只看切片也不够**（切片值平台根本不显示，评委看不到）。
下面第 1 条会展开讲。

> 另外先说清楚一点：我一开始从离线账本推断"平台部署的是 57.42 那个产物"，
> **这个推断是错的**，我修正了。**平台上根本没有 57.42** ——
> 平台 r3 是"未训练，保留父模型"。原因和这个修正带来的方法论后果，我在 8.1 末尾讲。

### 8.1 🔴 P0-1：C 方向平台实测比官方基线差 6.4%，且选型口径在平台上不可审计

**这次镜像在官方平台跑完的隐藏测试成绩**：

| 指标 | 平台实测值 |
|---|---:|
| **RMSE** | **56.25329 循环** |
| MAE | 39.938255 循环 |
| NASA 非对称评分 | 123 654 600 886.84648 |
| 隐藏测试样本 | 9 687 |
| 完成轮次 | 10 / 10 |
| 最终采用 | 第 5 轮候选 1 |
| CPU / 运行用时 | 7 859.56 s / 6 160.48 s |

**官方静态基线是 52.8686，所以这次是 56.25329 —— 差 6.40%。**

平台规则里有一条硬性要求：

> 要求：……**效果不低于平台基线**；……

**C 方向这条目前不满足**，直接影响客观评分的"平均性能提升 15 分"。

**下面是平台报告的逐轮公开验证指标**：

| 轮 | RMSE | NASA | 平台动作 |
|---:|---:|---:|---|
| 1 | 52.225263 | 5 290 473 776 | 采用本轮模型 |
| 2 | 52.225263 | 5 290 473 776 | 未训练，保留父模型 |
| 3 | 52.225263 | 5 290 473 776 | 未训练，保留父模型 |
| 4 | 51.153125 | 5 173 979 970 | 采用本轮模型 |
| 5 | **51.11116** | 4 790 681 137 | **采用本轮模型（最终）** |
| 6 | 51.11116 | 4 790 681 137 | 未训练，保留父模型 |
| 7 | **50.907431** | 4 791 636 526 | 未训练，保留父模型 |
| 8 | 50.907431 | 4 791 636 526 | 未训练，保留父模型 |
| 9 | 51.245666 | **14 008 639 868** | 未训练，保留父模型 |
| 10 | 51.245666 | 14 008 639 868 | 未训练，保留父模型 |

**从这张表能读出三件事：**

**① 你的核心担忧方向是对的 —— 公开验证确实低估了隐藏难度。**

最终采用的 r5 公开验证 **51.11116**，隐藏实测 **56.25329**，落差 **+10.06%**。
对照 B 方向：隐藏 609.08 **优于**公开 619.86。
**所以 `docs/C_WEIGHTING.md` §0 那条「评估口径和目标口径不一致」的判断，
平台数据是支持的**，这一点你的分析没做错。

**② 但"只看切片"这个解法在平台上有个硬伤：它不可见、不可审计。**

平台报告的"选中产物公开验证指标"列**只有全量 RMSE**，
切片值 `late_rmse` 平台根本不显示。
**一个评委看不到的口径做选型判据，等于没有判据。**

而且有个细节很能说明问题：
**r7 的公开验证 RMSE 50.907431 是全场最低，却被判"未训练，保留父模型"，
最终采用了 r5 的 51.11116。**
唯一合理的解释是选型确实用了某个非全量的口径（很可能就是 `selection_view`）——
**也就是说切片机制在平台上生效了**。但它没有让 C 变得比基线好。

**③ 搜索阶段那 104 条配置优化的 `proxy_rmse`，在平台上完全对不上。**

离线 proxy 28.50，平台隐藏 56.25，差 27.75。
这个指标平台不可见、不可验证，**用它做搜索目标等于在优化一个裁判看不到的量**。

**建议（都有现成落点）**：

1. **改成并列判据**：采纳条件 =「**全量不劣化**」**且**「切片改善」，
   而不是现在的"切片改善 + 全量允许劣化"。
   理由是平台数据两头都指向它：**只看全量不够**（r7 全量最低却未必最优），
   **只看切片不够**（不可审计）。
2. **`selection_guard()` 恢复 0.05**，并加**绝对下限**（全量劣化超基线 2% 一律不采纳）。
   guard 从 5% 放宽到 12% 那一版的 docstring 里写得很清楚
   （"12% is the bound that admits it"）—— 放宽的目的就是为了放行一个
   全量明显变差的候选。现在这个风险在平台上同样存在。
3. **把切片值写进平台可见的记录**（ledger / SwanLab / run_report），
   让"为什么 r7 没被采用、r5 被采用"**可复核**。现在这个决策在平台上是黑盒。
4. **把 `selection_view` 的口径定义写进技术报告**：
   它对应隐藏协议的什么假设、`horizon=200` 这个界从哪来、边界在哪。
   现在评委只能看到一个魔法数字。
5. **查清 C 的 10 轮为什么有 6 轮空转**（r2/r3/r6/r7/r8/r9/r10 全是"未训练，保留父模型"）。
   `the-test-is-truncated` 在平台 r3 根本没触发 —— 说明**离线成立的假设在平台上不成立**。
   轮次预算没用满是 C 落后的直接原因之一。
6. **`floor` 建议调回 0.20**（`88942e8` 那一版），那一版全量 **52.54**
   优于官方静态基线 52.87，是三次尝试里唯一全量优于基线的配置。

**另外三个平台观察**：

- **r9 的 NASA 异常跳变**：r7/r8 是 4.79e9，**r9 突然 1.40e10（2.9 倍）**，
  而 RMSE 只从 50.907431 变到 51.245666（+0.66%）。
  **RMSE 几乎没变而 NASA 涨 2.9 倍**，说明 r9 出现了少量预测值大幅偏离
  （可能预测了负 RUL 或极端值）。建议单独查一下 r9 的预测分布。
- **平台隐藏集 9 687 个样本 ≫ C-MAPSS 官方 test 的 707 台发动机**，
  说明隐藏集**不是"每机一行截断快照"**，而是多机多周期采样。
  这可能就是切片口径无法对齐隐藏协议的直接原因 ——
  值得据此重新审视 `horizon=200` 这个假设。
- **NASA 评分量级**：最终隐藏 NASA 1.2365e11，
  而按 `max(p,y)/10 - min(p,y)/10` 的常规定义，9687 样本最坏约 1.9e6，**差 5 万倍**。
  平台口径与我们代码里的 `nasa_asymmetric_score` 很可能不是同一个定义。
  它是"参考指标"不参与 RMSE 评分，但**技术报告里别拿它做结论**。

**最后一点，我要把话说清楚**：我原本从离线账本推断"平台部署的是 57.42 那个产物"，
**这个推断是错的** —— 平台 r3 是"未训练，保留父模型"，**平台上根本没有 57.42**。
平台与离线轨迹不同，是因为平台有真实 27B 决策层和真实漂移数据流。

**这条本身不是 bug，但它有方法论后果：离线 10 轮 ≠ 平台 10 轮。**
你那 104 条配置搜索、88 条实验记录同理 ——
**任何只用离线证据得出的结论，都不能直接当成平台结果。**
所以我上一轮建议"补一个离线对照 run"这条，**现在作废** ——
离线对照了也说明不了问题。要验证，只能用试运行（**还剩 8 次**）。

**另外两个连带的小问题**（同一个根因）：

- **`tools/run_harness_offline.py` 第 278–292 行的账本写出器是硬编码字段列表**，
  把 `RoundController.record()` 已经写好的 `selection_metric` / `selection_before` /
  `selection_after` / `selection_guard` / `reason` **全丢了**。
  我审查时就是因此**没法从账本看出第 3 轮为什么被采纳、回退了多少**，
  必须读代码才能反推 —— 这也是这个问题容易漏掉的原因。
  建议改成透传 entry 全部字段，并加一条断言：`validation_after` 存在时 `selection_after` 必须存在。
- **`docs/TECHNICAL_REPORT.md` 已过期**。报告写的是 11-21，C 加权这批提交是 11-22，
  而报告里明确说 "已从 B/C adapter 撤回"，但代码里 `THE_TEST_IS_TRUNCATED` 已经加进 C adapter
  并且 r3 被采纳了；§8 还写着"应在轮次记录里同时报告最后周期代理指标"，
  但这一版**改的是判定逻辑本身**，不只是"同时报告"。
  要么更新到新状态并把上面这条结论写进去，要么明确标注"本文描述 11-21 状态，
  C 加权口径重构见 `docs/C_WEIGHTING.md` §6"。**别让评委在两份互相矛盾的文档之间猜。**

### 8.2 🔴 P0-2：Dockerfile 把 pip 源默认写死成清华镜像

**位置**：`Dockerfile` 第 79–80 行

```dockerfile
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt
```

两个问题：

1. **`ARG` 的默认值直接生效。** 官方评测机如果访问不了这个源，
   **镜像构建直接失败**，整个提交报废。而当前 `requirements.txt` 只有
   `numpy>=1.26,<3`，走官方 PyPI 完全没问题 —— 也就是说这个改动**没有收益，只有风险**。
2. **`ARG` 声明在 SDK 安装之后**（第 79 行 vs 第 66 行），
   所以 `pip install /opt/contest-harness-sdk` 走官方源、numpy 走镜像源，
   **两条链路不一致**，这应该不是本意。

**建议**：默认值改回官方源，镜像源做成可选覆盖：

```dockerfile
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt
```

如果确实需要国内加速，用 `--extra-index-url` 而不是 `--index-url`，
这样镜像源挂了还能回落到官方源。

### 8.3 🔴 P0-3：README 头表报的是不可部署的配方

**位置**：`README.md` 第 37–39 行 + `artifacts/rmse_verification.json`

```
| **A** | 0.7240 | **0.9707** | **+34.07%** |
| **B** | 681.160 | **636.671** | **+6.54%** |
| **C** | 52.869 | **44.493** | **+15.84%** |
```

这三个数字来自 `rmse_verification.json`，其 recipe 里写着：

```json
"C": {"hidden": 128, "epochs": 200, "target": "cycles", "extra_features": "elapsed"}
```

`extra_features: "elapsed"` —— **"已运行循环数"**。隐藏请求拿不到，
因为平台预测请求只有 `{task, inputs}`，没有 cycle、没有 unit。

**这一点代码里自己就写明了**，`research/tracks/optimize_abc.py` 第 12–18 行：

> the **deployable** score (a model that only sees ``inputs`` at
> prediction time; no unit id, no elapsed count).
> The current research champions for B (two-stage) and C (``elapsed``
> feature) are **not deployable**: the platform's prediction request carries
> only ``{task, inputs}``.

`docs/TECHNICAL_REPORT.md` 也明确说了"不把不可部署的 389/44.49 当作提交成绩"，
而且它给的**可部署**数字是 A 0.7277 / B 639.33 / C 50.57。

**所以同一个仓库里两份文档给了两套矛盾口径**，而**评委先看 README**。

**建议**：头表改成可部署口径；研究口径另起一张表并明确标注"研究侧，不可作为提交成绩"；
头表下面加一行指向 `docs/TECHNICAL_REPORT.md` 的口径说明。

> **另外**：本地分支上我们已经把 B 侧的生命周期估计修好了（`fitted_lifetime`），
> 官方切分 RMSE 做到 **414.73**，**优于**这版 README 里的 636.671。
> 合并时 B 的数字需要重算 —— 否则会把已有的更好结果覆盖成更差的。

### 8.4 🟡 P1：三处

**P1-1　`_bounded_prior_weights` 在大 RUL 值上会 OOM 崩溃**
`contest_baselines/strategies.py` 第 97–137 行构造了一个 `(top+1)×(top+1)` 的**稠密核矩阵**：

| top | kernel 显存 |
|---:|---:|
| 362（C-MAPSS 实际最大 RUL） | 1.0 MB ✅ |
| 5 000 | 191 MB ⚠️ |
| 20 000 | 3.0 GB ❌ |
| 10⁹ | `numpy._core._exceptions._ArrayMemoryError` ❌ |

当前 C-MAPSS 最大 RUL 约 362，所以**现在不会炸**。但私有包是另一批数据，
RUL 量级不同就可能踩到，而且这是 `MemoryError` 不是优雅降级，**会直接把整轮打断**。
建议：换成**一维卷积**（`np.convolve` 或 `gaussian_filter1d`，核的对角带宽只有 8，数学结果不变），
或者对 `top` 夹一个上限并在夹断时记一条 `changed` 说明。

**P1-2　`metrics.delta()` 的 docstring 与实现不符**
`contest_baselines/layers/model/metrics.py` 第 107–116 行：docstring 说
"How much the **primary** metric moved"，实现返回的是 `selection_value` 差值。
引入 `selection_view` 之后两者可以不同，这个 docstring 现在是错的。
影响很小（`contest_baselines` 里没有调用点），但这是公开函数。

**P1-3　`enable_thinking=False` 的失败被误记为故障**
`harness/llm.py` 第 335 行，第一段降级请求带 `enable_thinking=False`。
如果 PyTRiO 对未知字段返 400，这会走 `LLMUnavailable` → 落到第二段"无参重试"。
**降级链设计是对的**，但第一段失败会计入 `failures`，把探测性参数的问题记成真故障。
建议第一段失败时**不**计 failure。

### 8.5 🟢 P2：几条小的

| 位置 | 问题 |
|---|---|
| `harness/llm.py` | 计划缓存的 key 含整个 `budget`（含 `tokens_left` 等随时间变化的量），缓存命中率会低于预期；建议只取决策相关子集 |
| `contest_baselines/strategies.py` | `the-tail-is-underweight` 与 `the-tail-is-underweighted` 两个 key 并存（用 `in (...)` 兼容），容易误用，建议统一成一个 |
| `docs/ABC_OPTIMIZATION.md` | A 的下一步写的是 `sample_l2`，`artifacts/tracks/a_transforms.json` 也产出了，但 A 的 adapter/strategies 没有对应改动；A 的 10 轮里 `more-data-helps` 连续 7 次被拒，建议明确 A 的落点 |
| 远端分支 | `origin/` 上有 `GY-ai+-2`(71 文件)、`copilot/merge-three-system-directions`(238)、`feat/minimal-harness`(240)、`feature/qwen-sft`(372)、`feature/task-a-engine`(97)，`main` 是唯一完整的一条；建议明确以 main 为准，其余归档或删除，避免后续合并踩坑 |

### 8.6 ✅ 我这边验证过、确认没问题的部分

也说一下核对过的结论，免得大家重复劳动：

- **全量测试全绿**：`359 passed, 1 skipped, 117 subtests passed`（Python 3.12，独立检出 `main` 跑的）
- **加权实现正确**：`sample_weight = ones` 时参数与 loss 与无权重路径**逐位相同**；
  加权梯度与所报 loss 解析一致；权重**不写入 checkpoint**，
  `Model.load()` 每轮新建对象 → **不跨轮泄漏**
- **`docs/TECHNICAL_REPORT.md` 的 10 轮数字与账本逐轮对得上**：
  A 0.714→0.7192、B 681.1598→456.4174、C 52.8686→50.6834，采纳/拒绝次数、最差候选、恢复轮次全对
- **模型 ID 命名已经统一了**：`harness/config/harness.json` 用短名 `Qwen3.8-27B`，
  `harness/llm.py` 的 `MODEL_ALIASES` 映射到 `Qwen/Qwen3.8-27B`，
  `train/sft_planner.py` 用平台名 —— 大小写不敏感、支持 `qwen3_8_27b` 等写法、
  支持 `<base>-ft-<date>` 派生模型
- **没有权重入库**：`git ls-files` 确认没有任何 `.pkl` / `.pt` / `.safetensors` 被跟踪
- **仓库卫生正常**：`MANIFEST.json` 已随新文件重新生成；
  `.gitignore` 正确排除 `train/data/`、`train/reports/`；
  `Dockerfile.research` 里**移除了** `COPY runs/ ./runs/`（研究镜像不该带运行产物）
- **27B 不在本地推理**，镜像不带权重

### 8.7 建议的处理顺序

1. **补关闭 selection_view 的 C 对照 run** → 定 P0-1
2. **改 Dockerfile 的 `PIP_INDEX_URL` 默认值** → 定 P0-2
3. **改 README 头表为可部署口径** → 定 P0-3
4. 账本写出器补字段 → P1-1（证据链）
5. 更新/标注 `TECHNICAL_REPORT.md` → P1-2
6. `_bounded_prior_weights` 换卷积或夹上限 → P1-3
7. 合并本地 B 侧的 `fitted_lifetime`，重算 B 的 README / `rmse_verification` 数字
8. `python tools/refresh_manifest.py` + `pytest tests -q` + `python -m tools.verify_repo`
9. 重建镜像做一次 4 核 / 8 GB 实跑

完整审查报告见本文件 §1–§7，每条都有可复现的证据位置。

**我这边暂时没有往 GitHub 推送任何东西**，等这 3 个 P0 处理完、验证跑过再一起提。

---

*报告结束。所有结论均基于 `origin/main` @ `4054a20` 的实际代码、账本与测试输出。*
