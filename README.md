# AI+ · 智训引擎（赛题 9）

工业制造场景下的模型训练与超参数优化 —— AI 驱动的工业模型自进化系统。

> 本仓库汇总三个赛道的实现，并保留原始分支，便于追溯各自的设计演进。
> 最新主线是 `three_track_system/`：**一个入口、一套闭环、三个数据集**的统一系统。

## 目录索引

| 路径 | 来源 | 说明 |
| --- | --- | --- |
| `three_track_system/` | `copilot/merge-three-system-directions` | **三方向统一闭环（最新主线）**：A 气体分类 / B 轴承 RUL / C 发动机 RUL，共享漂移检测、假设链、门控、回滚与证据链 |
| `three_track_system/README.md` | 同上 | 三方向成绩、零泄漏协议与复现命令 |
| `AI2/` | `GY-ai+-2` | **赛道 B 交付包（已同步到三方向版本）**：Harness 闭环 + 工业模型 + 证据链 + Docker 镜像 |
| `A_research_round09/` | `feature/task-a-engine` | **赛道 A（气体传感器漂移识别）** 引擎与 Round 09 复现配方 |
| `A_research_round09/README.md` | `feature/task-a-engine` | 赛道 A 详细说明（供团队成员阅读） |
| `cpu-c-guide-20261110-v1/` | `feat/minimal-harness` | **赛道 C（涡扇发动机 RUL）** 官方 CPU 参考基线包 |
| `harness/` | `feat/minimal-harness` | 最小可用闭环：漂移检测、诊断、实验、门控、上线/回滚 |
| `tools/` | `feat/minimal-harness` | 离线演示、漂移基准、镜像构建脚本 |
| `.github/workflows/` | 合并 | CI：编译、单测、账本校验、镜像往返校验 |

各赛道的详细说明请见：

- 三方向统一系统 → [`three_track_system/README.md`](three_track_system/README.md)，运行手册 → [`three_track_system/docs/RUNBOOK.md`](three_track_system/docs/RUNBOOK.md)
- 赛道 B → [`AI2/README.md`](AI2/README.md)，运行手册 → [`AI2/docs/RUNBOOK.md`](AI2/docs/RUNBOOK.md)
- 赛道 A → [`A_research_round09/README.md`](A_research_round09/README.md)、[`A_research_round09/REPORT.md`](A_research_round09/REPORT.md)、[`A_research_round09/API.md`](A_research_round09/API.md)
- 赛道 C → 官方基线包内的 `START.md` / `TASK.md` / `TRAINING_GUIDE.md`

## 三方向统一闭环成绩速览

| 方向 | 官方主指标 | 官方静态基线 | 智训引擎 | 相对提升 |
| --- | --- | ---: | ---: | ---: |
| A 气体分类（UCI Gas Sensor Array Drift） | Macro-F1 ↑ | 0.7240 | **0.9707** | **+34.07%** |
| B 轴承 RUL（XJTU-SY） | RMSE（分钟）↓ | 681.160 | **636.671** | **+6.54%** |
| C 发动机 RUL（NASA C-MAPSS） | RMSE（循环）↓ | 52.869 | **44.493** | **+15.84%** |

闭环的每一轮都在**从未参与训练**的留出单元（judge set）上打分，候选模型与冠军模型
在**同一批**留出单元上比较；验证集只在最后评一次分。数字来自
`three_track_system/runs/{A,B,C}/run_report.json`，可由
`python -m tracks.run --all --workspace runs --holdout-units 4 --judge-units 2` 复现。

## 赛道 B 成绩速览（AI2 交付包口径）

| | RMSE（分钟） | MAE（分钟） |
| --- | ---: | ---: |
| 官方静态基线 | 681.16 | 563.11 |
| **本系统闭环 champion** | **283.54** | **218.25** |

较官方基线 **−58.4%**；9 个决策周期，2 次 promote、1 次 reject、6 次 `do_nothing`，
3,457 参数。检查点格式 `contest.cpu-mlp.v1`，已实测可被官方 `baselines/model.py::Model.load()` 打开。

## 分支

| 分支 | 内容 |
| --- | --- |
| `main` | 合并后的主干 |
| `copilot/merge-three-system-directions` | 三方向统一闭环系统 |
| `GY-ai+-2` | 赛道 B 交付包 |
| `feature/task-a-engine` | 赛道 A 引擎 |
| `feat/minimal-harness` | 赛道 C 基线包与最小闭环 |

## 许可

见 [`LICENSE`](LICENSE)。
