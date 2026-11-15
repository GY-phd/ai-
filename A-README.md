# Task A｜气体传感器漂移识别与自适应更新

> AI+「智训引擎」比赛 · Task A  
> 当前交付：TaskAEngine v1 + Round 09 Replay Recipe  
> 当前状态：基础模块可运行，增量学习配方已实现，等待总 Harness 集成及正式评测验证。

---

# Part I｜项目说明（供团队成员阅读）

## 1. 我负责什么？

Task A 负责六类气体传感器数据的识别，以及传感器漂移条件下的模型维护和优化。

由于传感器会随运行时间、测量条件等因素发生响应变化，模型可能在新批次数据上出现误报或漏报。

我们的目标不只是训练一个识别准确率较高的模型，而是让系统能够：

**识别气体 → 监测漂移 → 分析问题 → 尝试优化 → 评估效果 → 决定上线或回滚。**

本模块负责其中的具体算法与模型操作，总 Harness 负责跨任务调度、Agent 决策和平台对接。

## 2. 目前完成了什么？

截至第九轮，已完成以下工作。

**基础模型与算法研究**

- 基于 128 维传感器特征识别六类气体。
- 完成九轮研究，涉及特征融合、乙醇与乙醛识别、异常检测、故障修复、结构化特征和持续学习。
- 发现公开验证数据中乙醇与乙醛存在明显的跨批次混淆。
- 多种候选算法在部分条件下改善了识别效果，但也暴露出跨批次误报与其他类别性能下降问题。

**模型管理模块**

已实现预测、诊断、候选生成、训练、评估、激活及回滚接口。

模型可以保留历史版本，未经必要验证不会自动替换当前稳定模型。

**第九轮增量学习扩展**

新增 `C_replay_anchor` 更新配方。

它能够在获得合法新标签后，结合历史数据约束更新模型，减少对旧类别识别能力的破坏。

该方法在部分合成漂移实验中表现良好，但尚未证明能稳定提升未知月份的真实分类性能。

## 3. 当前使用什么模型？

**默认活动模型：`stable`。**

这是目前保留的稳定版本。

新研究方法不会自动替换它：

- `raw_support` 等历史方案保留作为候选与对照。
- `structured_bootstrap` 用于初始化新的结构化分类头。
- `C_replay_anchor` 用于产生增量更新候选。

只有具备合法、独立的真实评估证据并通过检查后，才考虑激活新版本。

## 4. 当前工程结构

```text
task_a/
├── README.md
├── TaskAEngine_v1/
│   ├── task_a_engine/       # 原始 Engine 核心
│   ├── tests/               # 原 Engine 测试
│   └── README.md            # 原始交付说明
│
└── A_research_round09/
    ├── engine_recipe.py     # 新增 Engine 扩展
    ├── structured_model.py  # 结构化分类模型
    ├── recipe.py            # 增量学习更新公式
    ├── test_recipe.py       # 新增测试
    ├── API.md               # 详细对接说明
    └── REPORT.md            # 第九轮研究报告
```

上述为建议的 GitHub 交付结构；实际仓库如有调整，以当前目录为准。

## 5. 队友怎么对接？

总 Harness 应使用第九轮的 `RecipeEngine` 入口，以访问原 Engine 功能和新增增量学习配方。

主要操作包括：

| 接口 | 功能 |
|---|---|
| `predict` | 预测气体类别 |
| `diagnose` | 诊断输入分布变化 |
| `propose` | 提供候选更新方案 |
| `train` | 训练新候选 |
| `evaluate` | 比较新旧模型 |
| `activate` | 按条件切换活动模型 |
| `rollback` | 回滚到历史版本 |
| `status` | 查看当前模型状态 |

**注意：** A 模块不包含总 Harness 的 LLM 决策逻辑。总 Harness 需要根据 A 模块返回的诊断、实验指标和资源消耗，自行决定下一步操作。

具体参数和示例见 `A_research_round09/API.md`。

## 6. 当前存在的问题

- 公开 Batch 2 已被多轮研究使用，相关结果不能作为新的独立泛化验证。
- 乙醇与乙醛的跨月份混淆仍未得到可靠解决。
- 新增增量学习方法需要合法释放的新训练标签。
- 不同更新顺序、标签数量和类别缺失可能影响最终效果。
- 尚未完成正式比赛 SDK、实验记录平台及总 Harness 的完整集成验证。

## 7. 测试状态与下一步

目前交付记录显示：

- 原 Engine：16 项回归测试通过。
- 第九轮配方：8 项测试通过。
- 三阶段模拟流程已完成。
- 新增算法尚未通过真实独立的上线验证。

下一步优先完成团队接口集成、正式运行环境测试及实验记录对接，而非直接替换稳定模型。

---

# Part II｜AI Context / Agent Handoff

本部分面向 AI 编程助手、科研分析 Agent 和总 Harness 开发者。

以下字段用于明确项目结构、实验边界和维护约束。它们是当前工程上下文，不代表比赛官方 SDK 协议。

## PROJECT_IDENTITY

```yaml
project: AI+ 智训引擎
subtask: Task A
domain: Gas Sensor Drift Classification
input_dim: 128
sensor_structure: 16 sensors x 8 features
num_classes: 6
classes:
  - Ethanol
  - Ethylene
  - Ammonia
  - Acetaldehyde
  - Acetone
  - Toluene

default_active_model: stable
engine_version: TaskAEngine_v1
optional_extension: TaskAReplayRecipe_v1
extension_entry: A_research_round09/engine_recipe.py
extension_class: RecipeEngine
runtime_language: Python
minimum_python: "3.10"
primary_dependency: NumPy
```

## COMPONENT_RESPONSIBILITIES

**Task A Engine owns:**

- Gas classification and drift diagnostics.
- Candidate model training and prediction.
- Model evaluation and resource reporting.
- Checkpoint versioning, activation and rollback.
- Task A data-exposure and provenance checks.

**Master Harness owns:**

- Cross-task orchestration.
- LLM reasoning and final update decisions.
- Official SDK integration.
- Experiment tracking and reporting.
- Global resource budgeting.
- Verification of data-release permissions and evaluation independence.

Do not implement another master Harness inside Task A.

## SOURCE_OF_TRUTH

Read the following files before modifying the project:

```text
TaskAEngine_v1/README.md
TaskAEngine_v1/task_a_engine/engine.py

A_research_round09/API.md
A_research_round09/REPORT.md
A_research_round09/engine_recipe.py
A_research_round09/recipe.py
A_research_round09/structured_model.py
A_research_round09/test_recipe.py
```

When this README conflicts with implementation details, inspect the actual source code and tests. Report discrepancies rather than silently assuming either version is correct.

## DATA_AND_EVIDENCE

```yaml
public_training:
  name: Batch1
  samples: 445
  acetaldehyde_samples: 30

public_development_validation:
  name: Batch2
  samples: 1244
  acetaldehyde_samples: 109

historical_development_results:
  stable_macro_f1: 0.725843
  raw_support_macro_f1_approx: 0.8654

independent_future_generalization_verified: false
round09_activation_decision: no_update
```

Important interpretation:

- Batch 2 results are historical development results, not independent holdout evidence.
- Round 8–9 evaluations mainly use Batch 1 heuristic group splits and synthetic perturbations.
- Synthetic perturbation performance must not be reported as official competition performance.
- Do not use previously exposed evaluation data for new training or model selection.
- Do not assume unseen data are statistically independent solely because their IDs differ.

## MODEL_AND_TRAINING_STATE

```yaml
default_model: stable

available_base_candidates:
  - stable
  - fusion
  - witness_veto
  - raw_support
  - projected_support

optional_training_methods:
  - structured_bootstrap
  - C_replay_anchor

new_recipe_status: research_candidate
automatic_activation: false
```

`structured_bootstrap` fits a new structured classification head from authorized source data. It does not continue the original stable MLP weights.

`C_replay_anchor` performs incremental coefficient updates with constraints on historical model outputs.

Both methods create candidates without automatically changing the active model.

## INTEGRATION_CONTRACT

For the Round 09 extension, instantiate `RecipeEngine` rather than the original `Engine`.

Preserve the public interface:

```text
predict
diagnose
propose
train
evaluate
activate
rollback
status
```

The original Engine command-line entry point does not automatically support new checkpoint kinds introduced by the recipe extension.

Before implementing a bridge, read `A_research_round09/API.md`.

Training and evaluation require explicit legal data-release metadata.

A model may be trained without being activated.

The master Harness must separately authorize activation based on valid evidence.

## CRITICAL_RESEARCH_FINDINGS

1. Increasing acetaldehyde recall can simultaneously create serious false positives in ethanol and other gas classes.
2. Relative sensor-response features are not guaranteed to remain discriminative across months.
3. PCA reconstruction support does not guarantee classification correctness.
4. Hard sensor faults and ordinary distribution drift should not be treated as identical.
5. `C_replay_anchor` can recover performance under some simulated drift conditions, but performance is not monotonic with new-label count.
6. Incremental update order matters.
7. Improvements over a weak parent model do not establish superiority over the current `stable` model.
8. Existing evidence is insufficient to guarantee future cross-month gains.

## CHANGE_POLICY

When modifying Task A:

1. Preserve the original stable checkpoint and model assets.
2. Do not overwrite previous research runs or historical metrics.
3. Add experimental methods as separate candidates.
4. Keep training, calibration and evaluation data roles separate.
5. Run original Engine and extension regression tests.
6. Record per-class metrics, Macro-F1, Accuracy and resource cost.
7. Do not activate a candidate based only on synthetic or reused development results.
8. Keep changes scoped to Task A unless master Harness integration explicitly requires coordination.

## CURRENT_NEXT_ACTION

**Priority: integrate the existing Task A module into the team's master Harness.**

Required checks:

- Verify Python import paths and `RecipeEngine` integration.
- Verify official data formats and release permissions.
- Verify state persistence across evaluation stages.
- Verify experiment logging and resource reporting.
- Verify model activation and rollback in an authorized end-to-end scenario.
- Preserve `stable` as default until genuinely independent evidence supports a change.

Do not restart broad exploratory algorithm searches without a clearly defined scientific hypothesis or new evidence.

---

*Documentation status: Round 09 handoff. Update this file when the interface, default model or independent evidence status materially changes.*