# C_replay_anchor 可选配方接入说明

当前结论：**保留 stable 默认，C 配方仅生成待评估候选。** 本接口是 TaskAEngine_v1 的本地 Python 扩展，不是官方比赛SDK，也不承担总Harness或LLM Agent调度。

## 文件与安装

便携运行只需如下相邻布局；交付ZIP已包含原Engine及这些文件。

```text
TaskAReplayRecipe_v1/
  TaskAEngine_v1/                  # 原交付，原样保留
  A_research_round09/
    engine_recipe.py               # 七接口兼容的可选子类
    recipe.py                      # 固定C更新公式
    structured_model.py            # 从第八轮提取的相同特征/模型，无旧研究依赖
    mock_demo.py
    test_recipe.py
    requirements.txt
    API.md / REPORT.md / RESULT_TABLES.md / agent_decision.json
```

依赖只有NumPy；Python≥3.10，实测Python3.13.9、NumPy2.2.6。SQLite等来自标准库。代码路径基于文件位置解析，没有机器目录常量。默认沿用原单线程BLAS环境设置；如果总Harness先导入NumPy，应由它在进程启动前统一配置线程，重复训练的最后几位浮点数可能受运行库影响。检查点文件读取仍严格校验保存时SHA256。

从解压根目录安装依赖：

```powershell
python -m pip install -r A_research_round09/requirements.txt
python -X utf8 -B -m unittest discover -s A_research_round09 -p test_recipe.py -v
python -X utf8 -B A_research_round09/mock_demo.py --output mock_demo_new
```

Mock输出目录必须不存在，避免覆盖历史记录。三阶段依次为：六类12标签更新、缺乙醛12标签更新、无标签保留；每阶段实际诊断/评估，模拟激活均拒绝，最终stable。文件只用于验证接口流程，不提供识别精度证据。

## 初始化与七接口

```python
import sys
from pathlib import Path

delivery = Path("TaskAReplayRecipe_v1").resolve()  # 修改为实际解压目录
sys.path.insert(0, str(delivery / "A_research_round09"))
from engine_recipe import RecipeEngine

engine = RecipeEngine(store="a_model_store", mode="real")
print(engine.status())  # 新库默认stable；已有库保留已有状态
```

`predict`、`diagnose`、`propose`、`train`、`evaluate`、`activate`、`rollback`以及`status`沿用v1调用方式和返回格式。总Harness应在启用配方的进程中统一实例化 `RecipeEngine`：原 `Engine` 不认识新增检查点kind，原v1命令行JSON入口也不会自动加载本扩展。外部JSON调度器可以继续把 `action/args` 映射到这七个公开Python方法；不要暴露下划线开头的内部方法。此处没有改动旧CLI或全局Harness。

复用v1的内部持久化接口，因此本配方绑定交付的v1版本；升级Engine后应先运行回归测试。可复用现有v1版本库，但不要同时混用旧Engine实例访问已经激活的新kind。保存时一起备份整个store，包括SQLite和models目录。

## 无标签阶段

```python
diagnosis = engine.diagnose(unlabeled_features)
options = engine.propose(diagnosis)  # 只有no_update
predictions = engine.predict(unlabeled_features)
kept = engine.train(method="no_update")  # 不读取标签、不创建检查点
```

输入仍为N×128有限数值，类别顺序固定。诊断中的位移以初始化结构头时的原始特征均值/标准差为参考，实际模型采用冻结的结构特征中位数/IQR变换，两者用途不同。分数和熵未经过独立概率校准，不据此自动更新或确认漂移。

## 合法标签阶段：先创建候选

训练与评估JSONL沿用v1结构：每行含唯一`id`、`inputs.features`（128维）、`answer`（英文气体名）。类别为Ethanol、Ethylene、Ammonia、Acetaldehyde、Acetone、Toluene。

```python
source_release = {
    "purpose": "train", "labels": "released", "kind": "real",
    "release_id": "source-training-authorized"
}
new_release = {
    "purpose": "train", "labels": "released", "kind": "real",
    "release_id": "stage-training-authorized"
}

# 仅在没有结构头时执行一次。source必须合法、覆盖六类。
bootstrap = engine.train(
    "legal_source_train.jsonl", source_release,
    method="structured_bootstrap"
)
if bootstrap["status"] != "trained":
    raise RuntimeError(bootstrap)
parent_id = bootstrap["model_id"]

candidate = engine.train(
    "legal_stage_train.jsonl", new_release,
    parent_id=parent_id, method="C_replay_anchor",
    replay_path="legal_source_train.jsonl", replay_release=source_release
)
if candidate["status"] != "trained":
    raise RuntimeError(candidate)
candidate_id = candidate["model_id"]
candidate_predictions = engine.predict(unlabeled_features, model_id=candidate_id)
# 活动模型仍为stable；下一阶段可显式选择candidate_id继续生成后继候选。
```

注意：`structured_bootstrap` 是用合法源训练标签拟合固定A_hybrid头，不从stable MLP迁移权重。C只更新系数，特征变换和标准化固定；λ=1，没有可调参数，`parameters`必须为空。缺类标签允许训练，返回`class_counts`及`missing_calibration_classes`，不等于已证明能改善缺失类。

新增`train`参数：

| 参数 | 要求 |
|---|---|
| `method="structured_bootstrap"` | 合法源训练集六类齐全；生成新的结构头候选 |
| `method="C_replay_anchor"` | 明确选择结构头父版本；新校准至少1条；原数据读取器拒绝空文件 |
| `method="no_update"` | 不需要path/release；返回保留状态 |
| `replay_path/replay_release` | 源文件内容哈希必须与bootstrap绑定的源文件一致，覆盖六类，训练用途合法 |
| `parent_id` | C必须为该store中的结构头/后继；不会自动寻找“最好”父模型 |
| `parameters` | 新配方只接受None或空对象 |

保护规则：新校准与回放不能重叠；本继承链用过的校准不能再次计入；评估暴露不得转入此配方训练；已知Batch2拒绝。原v1其他训练方法的政策保持不变。身份检查涵盖ID及精确特征指纹，无法识别未知的数值变换副本、近重复或其他store中的历史，Harness应传递真实原始身份并维护跨模块暴露记录。

源回放在C中用于按源标签平衡的父输出约束，并非强制再拟合源真实标签。回放文件绑定到原始字节哈希，重新排版也会拒绝；保留原文件。参数/授权错误抛出异常；实际求解或保存失败返回`status="failed"`，原活动模型不变，并保留已发生的训练暴露记录。训练成功返回新检查点ID、SHA256、父版本、标签计数及CPU/墙钟耗时。

## 独立评估、启用与回滚

只能在候选和规则已经选定后，使用新释放且真正独立的评估文件：

```python
evaluation_release = {
    "purpose": "evaluate", "labels": "released", "kind": "real",
    "release_id": "fresh-heldout-authorized", "independent": True
}
evaluation = engine.evaluate(candidate_id, "fresh_independent_eval.jsonl", evaluation_release)
print(evaluation["agent_summary"])

# 总Harness还须依据授权、时间/测量组、类别覆盖及不确定性审核。
# 这里的harness_authorized是调用方自己的明确决策，不是模块自动生成。
if harness_authorized and evaluation["eligible"]:
    outcome = engine.activate(
        candidate_id, evaluation["evaluation_id"],
        expected_revision=evaluation["revision"]
    )
```

`independent=True`是调用方的事实声明，不是让数据变独立的开关。Batch1已用于种子训练，Batch2已用于开发，两者均不能作为新独立启用证据；改变ID、release_id或新建store不能正当化重复使用。所有评估都会登记暴露；不应在同一独立集上轮流试多个候选再选择。mock模式即使声明independent也不能激活此配方。

额外返回`recipe_enable_conditions`：real_store、independent_evidence、six_classes_evaluated。它们之外仍须满足原本的本地护栏（Macro-F1增益≥0.005，乙醛F1严格上升、precision≥0.65，其余类recall下降≤0.03），以及原版本/模型哈希一致性。`eligible`只说明必要条件通过，不是显著性检验；少量支持或组依赖明显时，应由Harness保留模型。本轮没有提出新的最低标签门槛。

如后续合法监控显示退化，可调用：

```python
state = engine.status()
rolled_back = engine.rollback(expected_revision=state["revision"])
```

没有历史激活时返回kept；有历史时校验旧检查点并回退。默认stable和其他四个种子方案一直保留。配方不会依据模拟压力分数自动切换，也不会自行调用LLM。

## 成本与精简输出

`train`返回资源成本包含拟合与检查点写入，标签文件解析和前置核验发生在该计时之前。`predict/diagnose`资源字段测的是模型计算，文件/数据库/加载属于额外接口开销。`evaluate`包含候选和当前模型各一次评分。需要总预算时，Harness应同时测量调用端到端耗时。

建议总Harness读取训练的`model_id/method/class_counts/resources`、评估的`agent_summary/recipe_enable_conditions/metrics/resources`，以及最终决策。不要把样本原文或完整事件账本发给Agent。所有模块资源返回的LLM calls/tokens为0；SwanLab和Agent Token由总Harness管理。

## 完整研究复现（原研究工作区）

运行时便携ZIP不需要历史研究代码；下面的科研复现还需要本地原有A_research_round08、A_acetaldehyde、任务A/baselines和合法Batch1数据。ZIP内的报告和原始结果可直接阅读。

```powershell
.\.venv\Scripts\python.exe -X utf8 -B A_research_round09/study.py --train 任务A/public/train.jsonl --reverse-checkpoint A_research_round08/run_01/reverse_month_A_hybrid.npz --output A_research_round09/run_new
.\.venv\Scripts\python.exe -X utf8 -B A_research_round09/benchmark.py --train 任务A/public/train.jsonl --run A_research_round09/run_01 --output A_research_round09/costs_new.json
.\.venv\Scripts\python.exe -X utf8 -B A_research_round09/verify.py --output A_research_round09/verification_new
.\.venv\Scripts\python.exe -X utf8 -B A_research_round09/summarize.py
```

输出使用新目录，避免覆盖结果。verify按已交付的run_01和run_replay_verified验证，另运行原Engine回归、配方测试及便携Mock；它需要原Engine ZIP和稳定检查点。报告中的实际验证记录位于verification/。科研复现没有Batch2输入参数。
