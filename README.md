# AI+ · 智训引擎（赛题 9）

工业制造场景下的模型训练与超参数优化 —— AI 驱动的工业模型自进化系统。

> 本仓库汇总了三个赛道/阶段的工作分支。原始分支全部保留，
> 便于追溯各自的设计演进。

## 目录索引

| 路径 | 来源分支 | 说明 |
| --- | --- | --- |
| `AI2/` | `GY-ai+-2` | **赛道 B（轴承 RUL）完整交付包**：Harness 闭环 + 工业模型 + 证据链 + Docker 镜像 |
| `A_research_round09/` | `feature/task-a-engine` | **赛道 A（气体传感器漂移识别）** 引擎与 Round 09 复现配方 |
| `cpu-c-guide-20261110-v1/` | `feat/minimal-harness` | **赛道 C（涡扇发动机 RUL）** 官方 CPU 参考基线包 |
| `harness/` | `feat/minimal-harness` | 最小可用闭环：漂移检测、诊断、实验、门控、上线/回滚 |
| `tools/` | `feat/minimal-harness` | 离线演示、漂移基准、镜像构建脚本 |
| `.github/workflows/` | 合并 | CI：编译、单测、账本校验、镜像往返校验 |

各赛道的详细说明请见：

- 赛道 B → [`AI2/README.md`](AI2/README.md)，运行手册 → [`AI2/docs/RUNBOOK.md`](AI2/docs/RUNBOOK.md)
- 赛道 A → [`A_research_round09/REPORT.md`](A_research_round09/REPORT.md)、[`A_research_round09/API.md`](A_research_round09/API.md)
- 赛道 C → [`赛题9规则：“智训引擎”—工业制造场景下的模型训练与超参数优化(2).pdf`](赛题9规则：“智训引擎”—工业制造场景下的模型训练与超参数优化(2).pdf)

## 赛道 B 成绩速览

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
| `GY-ai+-2` | 赛道 B 交付包 |
| `feature/task-a-engine` | 赛道 A 引擎 |
| `feat/minimal-harness` | 赛道 C 基线包与最小闭环 |
