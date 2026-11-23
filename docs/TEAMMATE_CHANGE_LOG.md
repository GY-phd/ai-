# 队友改动日志与内容清单（`origin/main` 相对本地基线）

> 基线：`26bbeed`（本地）　目标：`4054a20`（远端 main）
> 作者：**LI Yaopei** `<1002241223@email.cugb.edu.cn>`
> 改动时间：2026-11-22 14:31 → 19:51（单日连续 7 次提交）
> 总量：**+59 526 / −156** 行，70 个文件，**无文件被删除**
> 记录时间：2026-11-23

---

## 0. 先说三件容易搞错的事

**① 提交作者是 LI Yaopei，不是你。** 7 个提交全部署名 `LI Yaopei`，
`git log` 里可查。往 `main` 推的是队友的版本。

**② Adapter 插件化架构是**你**的成果，不是队友加的。**
本地基线 `26bbeed` 里 `contest_baselines/adapters/` 已经存在
（`base.py` / `hypotheses.py` / `profile_a.py` / `profile_b.py` / `profile_c.py` /
`registry.py` / `__init__.py` 七个文件都在）。
队友在这套架构上**只做了一次扩展**，往 `base.py` 加了 4 个方法（+43 行）：

```python
def test_horizon(self): ...      # 隐藏协议的剩余寿命上界，默认 None
def weight_horizon(self): ...    # 训练权重的目标窗口，默认跟随 test_horizon
def selection_view(self, ...): ...# 协议对齐的选型切片，默认 None
def selection_guard(self): ...   # 全量指标允许回退的比例，默认 0.10
```

**③ 第一个提交 `3c48f06` 是个"补提交"包。**
它的 commit message 自己写了「一并提交此前未入库的 A/B/C 协议审计、SFT planner、
分层重构等改动」—— 也就是说这个提交里的
`harness/llm.py`(+161)、`harness/config.py`(+14)、`train/sft_planner.py`(+296)、
`research/tracks/optimize_abc.py`(+298)、`docs/TECHNICAL_REPORT.md`(+236)
**都是更早写好的代码，只是这次才入库**，不是这两天新写的。

---

## 1. 提交逐条时间线

按时间正序，每条给出**提交说明 + 实际改了哪些文件 + 关键内容**。

### ① `3c48f06` — 14:31　重开 C 加权实验

```
新增 research/tracks/c_weight.py：对照、退化权重、经验/通用先验重要性
权重、截断目标等 30+ 变体，统一按公开验证全量与截断测试代理口径评估
新增 docs/C_WEIGHTING.md：记录结论（全量 RMSE 与隐藏代理背离；测试 RUL
分布加权把代理 RMSE 38.1→28.1，通用 uniform 先验 38.1→31.9 且全量不劣于
官方基线）
新增 artifacts/tracks/c_weight_*.json 与 research/data/cmapss/RUL_FD00x.txt
一并提交此前未入库的 A/B/C 协议审计、SFT planner、分层重构等改动
```

**实际改动**（+54 893 / −51，45 个文件）—— 大头是补提交：

| 文件 | 变化 | 性质 |
|---|---|---|
| `artifacts/tracks/c_weight_seeds.json` | +19 019 | 新证据（最大单文件） |
| `artifacts/tracks/c_weight_priors.json` | +12 700 | 新证据 |
| `artifacts/tracks/c_weight_screen.json` | +6 667 | 新证据 |
| `artifacts/tracks/c_weight_tempered.json` | +5 452 | 新证据 |
| `research/tracks/optimize_abc.py` | +298 | 补提交（A/B/C 协议审计） |
| `train/sft_planner.py` | +296 | 补提交（27B LoRA SFT） |
| `docs/TECHNICAL_REPORT.md` | +236 | 补提交 |
| `research/tracks/c_weight.py` | +338 | 新（加权实验框架） |
| `harness/llm.py` | +161 | 补提交（决策层加固） |
| `research/tracks/experiment_a_transforms.py` | +157 | 补提交（A 输入变换） |
| `contest_baselines/model.py` | +41 | 补提交（`sample_weight`） |
| `harness/config.py` | +14 | 补提交（`HARNESS_LLM_MODEL` 覆盖） |

**核心内容**：建立"隐藏协议代理口径"—— 每台公开验证发动机按其子集 `RUL_FD00x`
经验分布抽一个截断周期，取该发动机 `last_cycle − h` 那一行。
结论：**全量 RMSE 与隐藏代理显著背离**（`imp` 加权：全量 +17%，代理 −26%）。

---

### ② `4dd9aed` — 14:31　C 加权落地：协议对齐选型

```
新增 the-test-is-truncated 假设：均匀 RUL<=200 先验对训练密度的
sqrt 重要性权重，只改训练损失质量，C 适配器专属
MetricRecord 增加 selection 切片；采纳判决用 late_rmse，全量 RMSE
以 10% 回归上限作安全门
C 适配器提供 test_horizon/selection_view；ledger/SwanLab/review
同时记录 full 与 selection
补测试：权重行为、A 上跳过、selection 比较、full guard
离线 runner 保留每轮候选 checkpoint 便于复算
```

**实际改动**（+933 / −54，14 个文件）—— **这是核心架构改动**：

| 文件 | 变化 | 关键内容 |
|---|---:|---|
| `contest_baselines/layers/controller/round.py` | +165/−13 | `_measure()` 加 selection 切片；`decide()` 改用 selection 值；新增 `_full_split_guard()`；`record()` 写 guard/selection 字段；SwanLab 同时报两个指标 |
| `contest_baselines/layers/model/metrics.py` | +33 | `MetricRecord` 加 `selection_name` / `selection_value`；`compare()` 改比 selection |
| `contest_baselines/adapters/base.py` | +43 | 4 个新方法（见 §0②） |
| `contest_baselines/adapters/profile_c.py` | +38 | `test_horizon()`=200、`selection_view()`、`selection_guard()`=0.10 |
| `contest_baselines/adapters/hypotheses.py` | +19 | `TAIL_WEIGHTED`、`THE_TEST_IS_TRUNCATED` 两个假设 |
| `contest_baselines/strategies.py` | +58 | `_bounded_prior_weights()` 首次出现（floor=0.02）；两个新假设的 prepare 分支 |
| `tests/test_controller_round.py` | +49 | selection 比较 + full guard 测试 |
| `tests/test_strategies.py` | +45 | 权重行为测试 |
| `tests/test_layers.py` | +24 | MetricRecord 测试 |
| `tests/test_adapters.py` | +24 | 适配器测试 |
| `tools/run_harness_offline.py` | +9 | 保留每轮候选 checkpoint |

**关键数值**：`selection_guard()` 初始为 **0.10**（10%），
`the-test-is-truncated` 用 `horizon=200`（`weight_horizon` 尚未分离）。

---

### ③ `88942e8` — 14:32　C 加权校准：floor=0.20、**5% 全量门**

```
_bounded_prior_weights 的 floor 从 0.02 校准到 0.20：C 隐藏协议代理
RMSE 39.90→32.40（-18.8%），全量 52.87→52.54（低于静态基线）
the-test-is-truncated 提前到 C 假设序列第 3 位，epoch 预算 40；
6 轮离线路径在 r3 采纳，r2/r4/r5/r6 均被协议对齐选型拒绝
finish/submit/SwanLab 同时报告 late_rmse 与 full rmse
新增 runs/harness-offline-c-weight 6 轮证据
MANIFEST.json 重新生成（396 files, 6.3 MB, weights=False）
全量测试 341 passed, 17 skipped；本地 CI 9 项全 PASS
```

**实际改动**（+570 / −90，16 个文件）

**这一版是全量指标最好的一次**：floor=0.20 时全量 52.54，**低于静态基线 52.87**，
guard 也收到 5%。

| 文件 | 变化 |
|---|---:|
| `contest_baselines/strategies.py` | +13（floor 0.02→0.20） |
| `contest_baselines/adapters/profile_c.py` | +13（guard 0.10→**0.05**） |
| `contest_baselines/layers/controller/round.py` | +20 |
| `runs/harness-offline-c-weight/` | 新增 6 轮证据 |
| `MANIFEST.json` | +252/−… 重新生成 |

---

### ④ `61be8dc` — 17:40　C 加权参数搜索：H150 floor 0.15

```
60+ 配置搜索（形状/时域/floor/power/lr/epoch/逐子集/两阶段/cap）
生产选择 uniform H_train=150, floor=0.15, power=0.5；选型仍 H_sel=200
6 轮离线 C：r3 采纳，full 52.23→54.54（+4.4%，≤5% guard），
late 38.52→33.66，CMAPSS proxy 39.90→29.70（-25.6%）
适配器新增 weight_horizon()，与 test_horizon() 分离
文档记录两条 Pareto 前沿、三种子均值与取舍
343 passed, 17 skipped；本地 CI 9 项 PASS；MANIFEST 已刷新
```

**实际改动**（+317 / −175，17 个文件）—— **几乎全是搜索证据**：

| 文件 | 变化 | 说明 |
|---|---:|---|
| `artifacts/tracks/c_weight_search.json` | +1 996 | **86 条配置记录**，每条含四口径指标 |
| `research/tracks/c_weight_sweep.py` | +172 | 单配置复现入口（可复现证据） |
| `docs/C_WEIGHTING.md` | +117 | 两条 Pareto 前沿、三种子均值 |
| `MANIFEST.json` | +62 | 刷新 |

**架构变更**：`Adapter` 新增 `weight_horizon()`，与 `test_horizon()` **分离** ——
训练窗口 H_train=150 比选型窗口 H_sel=200 更窄。理由是"训练只需贴近隐藏截断分布，
选型仍要覆盖 RUL ≤ 200 的完整可疑区间"。

**这一版全量 54.54，guard 仍是 5%，+4.4% 在限内。**

---

### ⑤ `586be59` — 17:42　C 加权搜索证据：86 条配置记录

```
artifacts/tracks/c_weight_search.json：配置、四口径指标、最终 6 轮路径
research/tracks/c_weight_sweep.py：单配置复现入口
docs/C_WEIGHTING.md §7 指向产物与复现命令
MANIFEST 刷新；343 passed, 17 skipped
```

**实际改动**（+2 185 / −4，4 个文件）

补齐了 `c_weight_search.json` 的完整格式（四口径指标 + 最终路径）、
`c_weight_sweep.py` 的单配置复现入口、以及 `docs/C_WEIGHTING.md` §7 的复现指引。

---

### ⑥ `9c5ae50` — 18:09　C 加权最终 10 轮：H150 floor 0.05，**12% guard** ⚠️

```
10 轮离线路径 r3 采纳（full 52.23→57.42，+9.95% ≤12% guard），
r4-r10 全部拒绝；最终 production checkpoint = r3
最终 checkpoint 三口径：CMAPSS proxy 28.50（-28.6% vs 基线）、
uniform 33.66（-15.9%）、late≤195 32.85（-17.0%）、full 57.42
搜索产物更新为 104 条记录；文档、测试、MANIFEST 同步
343 passed, 17 skipped；本地 CI 9/9 PASS
```

**实际改动**（+890 / −211，20 个文件）

**这是最关键的一次改动 —— 它把 guard 从 5% 放宽到 12%，floor 从 0.15 降到 0.05。**

`contest_baselines/adapters/profile_c.py` 的实际 diff：

```diff
-        """C may give up at most 5% of the full-split RMSE.
-        The default 10% is too loose here: a late-life oversampling
-        candidate that improves the bounded slice by 12% drove the full
+        """C may give up at most 12% of the full-split RMSE.
+        H150/floor=0.05 weighting raises the full split by ~10% while
+        cutting the hidden-protocol proxy by ~28%, so 12% is the bound
+        that admits it without letting a candidate collapse.  The guard
-        return 0.05
+        return 0.12
```

`contest_baselines/strategies.py`：

```diff
-                           bandwidth=8.0, power=0.5, floor=0.15):
+                           bandwidth=8.0, power=0.5, floor=0.05):
-    2026-11-22 search selected ``horizon=150`` and ``floor=0.15``: on the
-    while the full-split RMSE is 54.5, still inside the round's 5% guard
+    2026-11-22 search selected ``horizon=150`` and ``floor=0.05``: on the
+    round's 12% guard over the parent (52.2).
```

**读法**：guard 从 5% 放宽到 12% 的**唯一目的**，按其 docstring 自述，
就是为了"admit it"（放行）那个把全量从 52.23 推到 57.42 的候选。
这不是疏忽，是**为了让收益数字好看而主动放宽了安全门**。

**参数演进轨迹**：

| 提交 | floor | guard | 全量 RMSE | 代理 proxy |
|---|---:|---:|---:|---:|
| `4dd9aed` | 0.02 | 10% | — | — |
| `88942e8` | 0.20 | **5%** | **52.54** ✅ 优于基线 | 32.40 |
| `61be8dc` | 0.15 | **5%** | 54.54 ✅ 在限内 | 29.70 |
| `9c5ae50` | **0.05** | **12%** | **57.42** ❌ 劣于基线 | **28.50** |

> 参数每次都往"更激进"的方向调**（floor ↓、guard ↑），
> 而**全量 RMSE 每次都往更差的方向走**（52.54 → 54.54 → 57.42），
> 代理 proxy 每次都更好（32.40 → 29.70 → 28.50）。
> **最后一次的产物是三版里唯一劣于官方静态基线的。**

### 🚨 平台实测（2026-11-23，队友把镜像提交官方平台跑完）

| 指标 | 平台实测值 | 对照官方静态基线 52.8686 |
|---|---:|---|
| **RMSE** | **56.25329 循环** | **+6.40% 劣于基线** |
| MAE | 39.938255 循环 | — |
| NASA 非对称评分 | 123 654 600 886.84648 | 量级异常大 |
| 隐藏测试样本 | **9 687** | ≫ C-MAPSS 官方 test 的 707 台发动机 |
| 完成轮次 | 10 / 10 | — |
| 最终采用 | 第 5 轮候选 1 | 离线账本里 r5 是**被拒**的 |
| CPU / 运行用时 | 7 859.56 s / 6 160.48 s | — |

**平台逐轮公开验证 RMSE vs 离线账本 full —— 轨迹完全不同：**

| 轮 | 平台公开验证 | 离线账本 full | 平台动作 |
|---:|---:|---:|---|
| 3 | **52.225263** | 57.4199 | 未训练，保留父模型 |
| 4 | 51.153125 | 57.4199 | 采用本轮模型 |
| 5 | **51.11116** | 57.4199 | 采用本轮模型（最终） |
| 7 | **50.907431** | 57.4199 | 未训练，保留父模型 |
| 9 | 51.245666 | 57.4199 | 未训练，保留父模型（NASA 突然 2.9 倍跳变） |

> **平台上根本没有 57.42 这个产物** —— 平台 r3 是"未训练，保留父模型"。
> **离线 10 轮 ≠ 平台 10 轮**（真实 27B 决策层 + 真实漂移数据流）。

**平台口径下的正确对照：**

| 口径 | 平台值 | 与隐藏 56.25 的差 | 说明 |
|---|---:|---:|---|
| 公开验证 RMSE（部署的 r5） | 51.111160 | **+10.06%** | 公开验证低估隐藏难度 |
| 最低公开验证 RMSE（r7，未采用） | 50.907431 | +10.51% | **全场最低却未被采用** |
| 官方静态基线 | 52.8686 | −3.38 | 比隐藏成绩还好 |
| 离线 `late_rmse`（切片） | 33.9105 | +22.34 | **平台不可见、不可审计** |
| 离线 `proxy_rmse`（截断代理） | 28.5025 | +27.75 | **平台不可见、不可验证** |

**三个推论**：

1. **公开验证确实低估隐藏难度**（C 落差 +10.06%，B 反而隐藏更好 10.78）
   → `docs/C_WEIGHTING.md` §0 的核心担忧**方向是对的**
2. **r7 全量最低却未被采用** → 选型确实用了某个非全量口径（很可能就是 `selection_view`），
   **说明只看全量不够**；但切片在平台不可见，**只看切片也不够**
   → 正确解法是**并列判据**（全量不劣化 **且** 切片改善）
3. **隐藏集 9 687 样本 ≫ 707 台发动机** → 平台隐藏集**不是"每机一行截断快照"**，
   `selection_view` 的协议模型前提**不成立**

**修正路径已有现成落点**：`88942e8` 那一版（floor=0.20、guard=5%、全量 **52.54**）
是三次尝试里唯一**全量优于官方基线**的配置。

---

### ⑦ `4054a20` — 19:51　作品文档：C 方向最终提交说明与镜像哈希

```
最终 C 配置（H150 f0.05、12% guard）与 10 轮结果
镜像 tar 路径、大小、SHA256 与平台字段
已知限制与备选方案
MANIFEST 刷新
```

**实际改动**（+169 / −2，2 个文件）—— 只加了一个文档：

| 文件 | 变化 |
|---|---:|
| `docs/作品文档.md` | +163（新增） |
| `MANIFEST.json` | 刷新 |

---

## 2. 改动内容分类清单

### 2.1 新增文件（41 个）

**证据产物（14）**
```
artifacts/tracks/a_transforms.json          +412
artifacts/tracks/c_weight_priors.json     +12700
artifacts/tracks/c_weight_scratch.json       +64
artifacts/tracks/c_weight_screen.json     +6667
artifacts/tracks/c_weight_screen2.json    +6195
artifacts/tracks/c_weight_search.json     +2462
artifacts/tracks/c_weight_seeds.json     +19019
artifacts/tracks/c_weight_soup.json         +358
artifacts/tracks/c_weight_tempered.json   +5452
artifacts/tracks/opt_abc.json               +635
artifacts/tracks/opt_abc_a.json             +270
artifacts/tracks/opt_abc_b.json             +165
artifacts/tracks/opt_abc_c.json             +203
```

**文档（6）**
```
docs/ABC_OPTIMIZATION.md        +206
docs/C_WEIGHTING.md             +285   ← C 加权主文档
docs/INDUSTRIAL_CPU_CHECK.md    +143
docs/PLANNER_SFT_RECORD.md      +108   ← 27B SFT 记录
docs/TECHNICAL_REPORT.md        +236
docs/作品文档.md                 +163   ← 最终提交说明
```

**研究脚本（4）**
```
research/tracks/c_weight.py              +338
research/tracks/c_weight_sweep.py         +172
research/tracks/experiment_a_transforms.py+157
research/tracks/optimize_abc.py           +298
```

**SFT（6）**
```
train/__init__.py            +8
train/README.md             +144
train/build_planner_sft.py  +433
train/eval_planner.py       +131
train/sft_planner.py        +296
tests/test_llm_planner.py   +114
```

**离线证据（12）**
```
runs/harness-offline-c-weight/ledger.jsonl
runs/harness-offline-c-weight/run_report.json
runs/harness-offline-c-weight/swanlab/C-round-1..10.jsonl
```

### 2.2 修改文件（29 个）

| 类别 | 文件 | 改了什么 |
|---|---|---|
| **选择协议** | `layers/controller/round.py` (+165/−13) | selection 切片判定、full guard、SwanLab 双指标 |
| | `layers/model/metrics.py` (+33) | `MetricRecord` 加 selection；`compare()` 改比 selection |
| | `adapters/base.py` (+43) | 4 个新方法 |
| | `adapters/profile_c.py` (+38→+63) | `test_horizon`/`weight_horizon`/`selection_view`/`selection_guard` |
| | `adapters/hypotheses.py` (+16→+35) | 2 个新假设 |
| **训练** | `contest_baselines/strategies.py` (+11→+98) | `_bounded_prior_weights` + 2 个 prepare 分支 + epoch 预算 |
| | `contest_baselines/model.py` (+41) | `fit()` 支持 `sample_weight` |
| | `rulkit/model.py` (+41) | `fit()` 支持 `sample_weight` |
| | `rulkit/rul.py` (+5) | `LifeFractionModel.fit()` 透传 `sample_weight` |
| **27B 决策层** | `harness/llm.py` (+161) | 探测缓存、计划缓存、`compact_history()`、三段降级、`<think>` 剥离 |
| | `harness/config.py` (+14) | `HARNESS_LLM_MODEL` 环境覆盖 + 新配置项 |
| | `harness/config/harness.json` (+7) | 同上 |
| | `harness/orchestrator.py` (+20) | 预算耗尽早退、planner_status 扩展 |
| | `harness/run.py` (+18) | 固定 BLAS 线程数 |
| **构建** | `Dockerfile` (+6) | ⚠️ pip 镜像源（见审查报告 P0-2） |
| | `Dockerfile.research` (+7) | 移除 `COPY runs/` |
| | `.gitignore` (+4) | 排除 `train/data/`、`train/reports/` |
| | `run_harness.sh` / `run_harness.ps1` | 线程数设置 |
| | `MANIFEST.json` (+280/−…) | 每次提交都刷新 |
| | `tools/run_harness_offline.py` (+9) | 保留候选 checkpoint |
| | `tools/verify_repo.py` (+5) | |
| **测试** | `test_adapters.py` / `test_config.py` / `test_controller_round.py` / `test_layers.py` / `test_sample_weight.py` / `test_strategies.py` | 新增 +139 行 |

### 2.3 未改动但**存在隐患**的部分

| 文件 | 隐患 |
|---|---|
| `contest_baselines/layers/controller/round.py` | `record()` 已写 `selection_guard` 等字段，但 `tools/run_harness_offline.py` 的账本写出器硬编码字段列表，**把它丢了** |

---

## 3. 一张时间线看懂整件事

```
14:31  3c48f06  建立代理口径 + 补提交一堆早写好的代码
14:31  4dd9aed  selection_view 上线，guard=10%
14:32  88942e8  floor=0.20，guard=5%   → 全量 52.54 优于基线 ✅
17:40  61be8dc  86 条配置搜索，floor=0.15，guard=5% → 全量 54.54 ✅
17:42  586be59  补齐搜索产物 + 复现入口
18:09  9c5ae50  floor=0.05，guard=12%  → 全量 57.42 劣于基线 ❌
19:51  4054a20  最终提交文档
```

**参数每次都往"更激进"的方向调**（floor ↓、guard ↑），
而**全量 RMSE 每次都往更差的方向走**（52.54 → 54.54 → 57.42），
代理 proxy 每次都更好（32.40 → 29.70 → 28.50）。
**最后一次的产物是三版里唯一劣于官方基线的。**

---

## 4. 值得肯定的部分

- **commit message 写得极好**：每条都带具体数字、具体理由、具体测试结果
  （"343 passed, 17 skipped；本地 CI 9/9 PASS"）。这种记录密度在竞赛项目里少见。
- **搜索证据完整且可复现**：104 条配置记录 + `c_weight_sweep.py` 单配置复现入口，
  结论可被第三方重跑验证。
- **负结果如实记录**：`docs/TECHNICAL_REPORT.md` 里"不把不可部署的 389/44.49
  当作提交成绩"、"B 的晚期加权无效（639.33→642.16）"这类诚实记录很有价值。
- **架构扩展克制**：往 `Adapter` 只加了 4 个方法，没有推翻你原有的抽象。
- **测试同步补齐**：新增 139 行测试，覆盖权重行为、selection 比较、full guard。
- **`sample_weight` 实现正确**：`weights=ones` 时与无权重路径逐位相同，
  不写入 checkpoint，不跨轮泄漏。

---

## 5. 需要跟进的部分

详见 `docs/GITHUB_VERSION_REVIEW.md` §2–§4。要点：

| 级别 | 问题 | 位置 |
|---|---|---|
| 🔴 P0-1 | 最终部署的 C 模型 57.42，**劣于官方基线 52.87**；guard 从 5% 放宽到 12% 正是为了放行它 | `9c5ae50` / `runs/harness-offline-c-weight/` |
| 🔴 P0-2 | pip 源默认写死清华镜像，评测机访问不了则**镜像构建失败** | `Dockerfile` 79–80 |
| 🔴 P0-3 | README 头表用含 `elapsed` 的不可部署配方报成绩 | `README.md` 37–39 |
| 🟡 P1-1 | 离线账本丢掉 `selection_guard`/`reason`，门控证据不可见 | `tools/run_harness_offline.py` 278–292 |
| 🟡 P1-2 | `TECHNICAL_REPORT.md` 写的是 11-21 状态，与 11-22 的代码矛盾 | `docs/TECHNICAL_REPORT.md` |
| 🟡 P1-3 | `_bounded_prior_weights` 稠密核在大 RUL 值上 OOM | `strategies.py` 97–137 |
| 🟢 P1-4 | `metrics.delta()` docstring 与实现不符 | `metrics.py` 107–116 |

---

## 6. 复现命令

```powershell
# 完整日志（带日期和作者）
git log 26bbeed..origin/main --date=format:"%Y-%m-%d %H:%M" `
  --pretty=format:"%h | %ad | %an%n%s%n%b%n---"

# 单个提交的完整 diff
git show 9c5ae50

# 只看某个文件的演进
git log -p 26bbeed..origin/main -- contest_baselines/adapters/profile_c.py

# 新增/删除/修改文件清单
git diff --diff-filter=A --name-only 26bbeed origin/main
git diff --diff-filter=D --name-only 26bbeed origin/main
git diff --diff-filter=M --name-only 26bbeed origin/main

# 对比某参数的演进
foreach ($c in @("4dd9aed","88942e8","61be8dc","9c5ae50")) {
  $v = git show "${c}:contest_baselines/adapters/profile_c.py" 2>$null
  $i = ($v | Select-String "def selection_guard").LineNumber
  "{0} -> {1}" -f $c, (($v | Select-Object -Skip $i | Select-String "return [0-9.]+" |
    Select-Object -First 1).Line.Trim())
}
```

---

*清单结束。所有行数、文件数与引文均取自 `git log` / `git show` 的实际输出。*
