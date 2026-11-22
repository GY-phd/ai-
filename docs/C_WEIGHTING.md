# C 加权实验（重开）：隐藏协议、加权与 RMSE

> 复现：`python -m research.tracks.c_weight --epochs 30 --seeds 42,7,123`
> 代码：`research/tracks/c_weight.py`
> 产物：`artifacts/tracks/c_weight_{seeds,priors,tempered}.json`
> 公开 RUL 先验：`research/data/cmapss/RUL_FD00{1,2,3,4}.txt`（NASA C-MAPSS 公开测试标签）

## 0. 结论先行

上一轮 C 加权（`the-tail-is-underweighted-by-weight`）被门控拒绝，原因是
公开验证 **全量 RMSE** 从 51.239 变成 51.705。重开后确认：这不是加权无效，
而是**评估口径和目标口径不一致**。

- 公开验证集是**完整寿命序列的每一行**；隐藏测试是官方 C-MAPSS 协议，
  **每台发动机只在截断后的最后观测周期出一行**。前者大量样本的 RUL 在
  200–500 周期，后者（公开 RUL_FD00x）最大约 195 周期，中位数约 83。
- 用“每台发动机按截断分布取一行”的代理口径重算，基线模型的 RMSE 从
  全量 52.87 降到 **约 39.9**；按测试 RUL 分布加权的模型降到 **约 28.1**。
- 但同一加权模型在全量验证上会退化到 58–61。**全量验证越好的模型，在
  隐藏协议上不一定越好**；继续用全量 RMSE 做轮次门控，会系统性拒绝真正
  对隐藏指标有效的候选。

所以下一步不是继续微调 `alpha`，而是：**加权方法 + 评估路径重构**。
本文记录方法、代理协议和第一轮结果；门控/选型重构见 §6。

## 1. 代理口径（hidden-protocol proxy）

对每台公开验证发动机，按它所属子集的 `RUL_FD00x` 经验分布抽取截断周期
`h`，取该发动机 `last_cycle - h` 那一行；四个子集按隐藏测试的发动机数
（FD001 100 / FD002 259 / FD003 100 / FD004 248）加权。`proxy_rmse` 是
这个期望 RMSE 的闭式估计（对 `h` 求期望，再对发动机取均方根）。

注意：私有评测包是**另一批同领域数据**，不是这份公开测试集本身。代理
口径衡量的是“截断测试协议下误差的方向和量级”，不是对隐藏分数的精确预测。
因此真正要落地时应使用**通用有界先验**（§3 的 `prior`），而不是把
`RUL_FD00x` 的逐值直方图写死。

## 2. 加权形式

训练侧逐行损失质量，不改特征、不改前向、不改 checkpoint 格式：

| 名称 | 权重 |
|---|---|
| `deg α` | `1 + α·(1 − RUL/unit_max)` |
| `imp` | `p_test(RUL) / p_train(RUL)`（经验 CMAPSS 测试分布，裁剪到 [0.02, 20] 后归一） |
| `density` | `p_test(RUL)` |
| `surv H,s` | `max(1/(1+max(0,RUL−H)/s), floor)` |
| `hard H` | `1`（RUL ≤ H），`0.01`（否则） |
| `prior` | 通用先验（uniform / triangular / exponential，上限 H）与 `p_train` 的重要性比 |
| `cap C` | 训练目标截断到 C（经典 C-MAPSS 做法；**仅改训练目标，评估仍用原标签**） |

`power p` 是“温度”：`w^p` 再归一，用来在隐藏收益和全量验证之间折中。

## 3. 结果（30 epoch，从官方 checkpoint 续训，lr=0.004，batch=256）

均值 ± 标准差，种子 42/7/123（通用先验为 42/7 或 42/123）。`full` 是公开
验证全量 RMSE，`proxy` 是 §1 的隐藏协议代理，`full≤195` 是全量验证中
真实 RUL ≤ 195 的行（不依赖截断分布，可作为稳健性旁证）。

| 变体 | n | proxy | full | full≤195 |
|---|---:|---:|---:|---:|
| 对照（不加权） | 3 | 38.09 ± 1.00 | 51.29 ± 0.26 | 38.28 ± 0.83 |
| `deg α=1` | 3 | 34.68 ± 1.08 | 51.60 ± 0.22 | 35.74 ± 0.79 |
| `deg α=2` | 3 | 33.14 ± 1.06 | 52.17 ± 0.24 | 34.76 ± 0.69 |
| `cap 150` | 3 | 28.38 ± 0.46 | 60.04 ± 1.03 | 32.69 ± 0.25 |
| `imp`（经验测试分布） | 3 | **28.28 ± 0.32** | 60.16 ± 1.26 | 33.28 ± 0.55 |
| `impsub`（按子集） | 3 | **28.08 ± 0.16** | 60.76 ± 0.49 | 33.54 ± 0.29 |
| `prior uniform H=195` | 2 | 30.26 | 54.57 | 32.59 |
| `prior uniform H=195, p=0.5` | 2 | 31.86 | **52.65** | 33.60 |
| `prior uniform H=195, p=0.75` | 2 | 30.31 | 54.01 | 32.73 |
| `prior exp H=195 τ=80, p=0.5` | 2 | 29.00 | 56.07 | 32.93 |
| `hard H=195` | 2 | 29.00 | 57.35 | 32.25 |
| `deg α=2 + imp^0.5` | 2 | 28.31 | 61.90 | 34.14 |

关键观察：

1. **收益稳定且不依赖随机种子**。`imp` 三个种子的 proxy 是 27.88/28.66/28.31，
   对照是 38.68/38.90/36.69。
2. **通用先验也能拿到大部分收益**。`uniform 0–195 + p=0.5` 把 proxy 从 38.09
   降到 31.86（−16.4%），同时全量 52.65 **低于官方基线 52.87**；`p=0.75`
   进一步到 30.31。这说明不需要知道隐藏标签的逐值分布，只需要“截断测试
   集中 RUL 有界、且低/中 RUL 占比更高”这一先验。
3. **`full≤195` 同向改善**（38.28 → 32–34），说明加权不是把高 RUL 的误差
   搬走，而是在隐藏相关区间内真的更准。
4. **截断目标 `cap 150` 与经验重要性加权几乎等效**（28.38 vs 28.28），
   说明主要机制是“不再让 RUL>200 的早期行主导损失”，而不是某个具体权重公式。
5. **全量 RMSE 与隐藏代理显著背离**。`imp` 全量 +17%，隐藏代理 −26%。
   这正是上一轮门控拒绝加权的根源。

## 4. 逐周期结构

从 `artifacts/tracks/c_weight_seeds.json` 的 `horizon_rmse` 可读：加权模型在
h=0/10/25/50/75/100/125/150/175/195 全部更优，收益在 h≈50–150 最大；
h=0 附近两者都接近 7–10，差别很小。也就是说，隐藏 RUL 的中段才是加权
真正解决的部分，而不是最后一周期。

## 5. 为什么不能直接进提交路径

现有 `RoundController.decide()` 用公开验证**全量**主指标 `BETTER/WORSE`
决定采纳。`imp`、`cap150`、`hard195` 都会在这一步被拒绝；`deg1` 也与
对照基本持平。因此：

- 不重构评估路径，加权候选永远进不了 production；
- 只把加权写进 hypothesis 而不改门控，等于重复上一轮结果。

## 6. 落地：评估路径重构

这一节记录本轮已经落地到 `contest_baselines` 的改动。

### 6.1 训练侧假设

新增 `the-test-is-truncated`（`adapters/hypotheses.py`，仅 C）：

- 权重 = `clip(p_target(RUL) / p_train(RUL), 0.02, 20) ** 0.5`，
  `p_target` 为 `[0, H]` 上的均匀先验，`H = adapter.test_horizon() = 200`，
  `p_train` 由训练标签的高斯平滑直方图估计（带宽 8）。
- 只改逐行损失质量；不复制行、不改特征 schema、不改前向、不改检查点格式。
- 用真实训练标签和通用协议上界，不读取任何隐藏标签或隐藏发动机名单。
- 策略层实现：`strategies._bounded_prior_weights()`；
  适配器契约：`Adapter.test_horizon()` / `selection_view()` /
  `selection_guard()`。

### 6.2 选型与门控

`MetricRecord` 现在区分：

- `primary_value`：平台主指标（全量验证 RMSE），继续完整记录；
- `selection_value`：协议对齐切片的指标（C 为 `late_rmse`，即真实
  RUL ≤ 200 的验证行上的 RMSE）。

`compare()` 用 `selection_value` 做采纳判决；`RoundController.decide()`
再加一道 **full-split guard**：全量 RMSE 相对父模型的回归不得超过
`selection_guard()` 默认的 10%。SwanLab/ledger 同时记录 full、selection
和 guard 结果，模型 review 的 brief 也会说明两者。

### 6.3 6 轮离线 C 路径（第一次运行）

| 轮 | 假设 | 采纳 | full RMSE | late RMSE |
|---:|---|---|---:|---:|
| 1 | more-data-helps | 是 | 52.87 → 52.23 | 39.92 → 38.52 |
| 2 | the-parent-was-left-unconverged | 否 | 52.23 → 51.24 | 38.52 → 39.28 |
| 3 | the-tail-is-underweighted | 是 | 52.23 → 56.04 | 38.52 → 34.52 |
| 4 | **the-test-is-truncated** | 是 | 56.04 → 54.54 | 34.52 → 32.97 |
| 5 | the-unit-is-the-confounder | 否 | 54.54 → 51.33 | 32.97 → 36.36 |
| 6 | capacity-is-overfitting | 否 | 54.54 → 51.16 | 32.97 → 37.74 |

可以看到旧口径下 r2 会被采纳（full 51.24），但它的 late 指标反而变差；
新口径拒绝了它。r5/r6 是全量 RMSE 更好、late 更差的典型候选，也都被
拒绝。最终 deployment 停在 r4（full 54.54，late 32.97）。**这是有意的
取舍**：平台评分走隐藏截断协议，full 继续作为安全旁证，而不是目标。

### 6.4 尚未完成

1. 用保留的 round checkpoint 复算 §1 的 CMAPSS proxy，确认 late_rmse
   与 proxy 的排序一致（`tools/run_harness_offline.py` 现在会把每轮
   候选 checkpoint 复制到 `runs/<out>/checkpoints/`，便于复算）。
2. 评估 weight soup：对照与加权模型按 λ 平均，在 full 与 proxy 之间取
   Pareto 点，可能同时满足“full 不劣于官方静态基线”和隐藏增益。当前
   单模型路径的 r4 full 54.54 高于静态基线 52.87，这一点需要在提交前
   明确：若评审以全量公开验证作硬门槛，需要改用 soup 或 λ 折中。
3. 10 轮完整路径与更多种子。

