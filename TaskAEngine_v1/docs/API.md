# Task A Engine Python / JSON 接口 v1

本接口供团队总 Harness 使用，不代表官方比赛 SDK。总 Harness 负责阶段组织、数据及标签授权、Agent 决策、SwanLab 和总预算；A Engine 负责确定性模型计算、评估记录及本地版本管理。接口均同步执行。

## 1. 初始化、持久化与类别

```python
from task_a_engine import Engine
engine = Engine(store, mode="real", policy=None)
state = engine.status()
```

`store` 是调用者可写目录，首次自动创建，默认启用 `stable`、`revision=0`。其中 `engine.sqlite3` 保存模型注册、数据使用、评估及切换记录，`models/` 保存新检查点。运行时无固定工作目录要求，包自带只读模型资产。移动工程后，保留完整包与整个版本库即可；不要只复制 SQLite 或部分检查点。

`mode` 为 `real` 或 `mock`，创建后不可改变。种子资产清单、数据指纹清单和策略也固定到版本库，重新打开时校验；不应原地替换种子资产后沿用旧版本库。`policy=None` 在新库使用默认值，在已有库沿用保存值。`status()` 返回 `active_model_id`、`revision`、`mode` 和模型列表（`model_id`、`kind`、`sha256`），不返回原始样本。

六类顺序固定，`scores` 的列与 `classes` 一致：

| 索引 | 类别 | 中文 |
| --- | --- | --- |
| 0 | `Ethanol` | 乙醇 |
| 1 | `Ethylene` | 乙烯 |
| 2 | `Ammonia` | 氨 |
| 3 | `Acetaldehyde` | 乙醛 |
| 4 | `Acetone` | 丙酮 |
| 5 | `Toluene` | 甲苯 |

## 2. 输入与标签释放声明

每个样本必须是按原始编号 1–128 排列的 128 个有限数值。保持原始尺度，不要再次标准化，也不要添加阶段或 ID 特征。布尔值、NaN、Infinity、空批次和错误维度会被拒绝。

`predict` / `diagnose` 的 `inputs` 接受：

- 一维长度 128 数组，或 `N×128` 列表 / NumPy 数组；
- `{"features": [...]}`，或 `{"id":"sample-1","inputs":{"features":[...]}}`，以及后一种对象列表；
- JSONL 文件路径。文件中的每行必须有非空字符串 `id` 与 `inputs.features`。

`inputs` 对象内部只允许 `features`。同一批次中已提供的 ID 必须唯一。文件按 UTF-8（可带 BOM）解析，空行忽略，最大 128 MiB；重复 JSON 字段拒绝。文件级用途由接口指定，文件中额外的 `answer` 不会被预测或诊断使用。

`train` / `evaluate` 只接受 JSONL 文件路径，每行必须额外提供英文类别 `answer`。下例中的省略号是说明，真实文件必须填满 128 个数值：

```text
{"id":"batch-new-0001","inputs":{"features":[128 个数值]},"answer":"Ethanol"}
```

两个标签用途分别声明，不能互换：

```python
training_release = {
    "purpose": "train", "labels": "released", "release_id": "authorized-training-release-01",
    "kind": "real"
}
evaluation_release = {
    "purpose": "evaluate", "labels": "released", "release_id": "held-out-release-01",
    "kind": "real", "independent": True
}
```

`release_id` 必须非空；`kind` 必须与版本库模式一致。评估必须显式声明布尔 `independent`；训练不能声明 `independent=True`。释放声明是上游调用者的授权凭据描述，不会向官方平台查验权限，也不能证明统计独立性。Harness 应保留其真实授权来源及分组/时间划分依据。

数据隔离使用文件哈希、样本 ID 哈希及规范化 float64 特征行哈希。仅修改 ID 或 JSON 排版不能让训练样本成为独立评估样本。任一行与本库已训练数据重叠会拒绝整个评估；任一行与本库已评估数据重叠会让整份新评估失去独立资格。原 Batch 1 的训练暴露及公开 Batch 2 的开发暴露已预置，已知 Batch 2 通过文件哈希、行指纹、`gas_b02_` ID 前缀识别，标签禁止训练。

因此，Harness 应在看到评估标签和结果之前，选定一个候选用于一次独立评估。相同数据集上第二次评估另一个候选会被标记 `previously_evaluated=true`，即使传入新的 release_id 或新的样本 ID，也不能支持启用。v1 没有多候选比较的统计校正，也不提供批量盲测 API；已有一次独立评估证据本身仍可按 `activate` 规则使用。

其他真实评估数据以后获得合法训练授权时可用于训练，但其旧评估不能再支持启用。行指纹不能发现数值被人为扰动、未登记的外部训练、隐藏的跨组依赖或其他版本库暴露；这些由上游数据管理保证。不要通过创建新库绕过暴露记录。

## 3. 七个接口

### `predict(inputs, model_id=None)`

使用指定模型，省略时使用调用开始时读取到的当前模型。返回 `model_id`、当前版本库 `revision`、`classes`、每行 `labels`、`N×6` 的 `scores`、`resources`。指定历史模型时，`revision` 仍指当前活动状态，不是该历史模型的版本号。分数是模型归一化输出，不代表经过独立校准的可靠概率。本调用不更新模型或数据使用记录。

### `diagnose(inputs, model_id=None)`

返回紧凑的无标签诊断：

| 字段 | 含义 |
| --- | --- |
| `model_id`, `revision`, `rows` | 本次模型、当前状态和输入数 |
| `reference` | `parent_training_scaler`，沿父模型找到的 MLP 训练均值与标准差 |
| `mean_shift_l2` | 标准化后特征均值的 L2 范数 |
| `outside_3sd_fraction` | 绝对标准化值超过 3 的元素占比 |
| `top_sensors` | 位移最大的 3 个传感器（1–16，每个 8 特征） |
| `predicted_counts` | 六类预测数量 |
| `mean_entropy` | 归一化分数的平均熵，自然对数 |
| `labels_used` | 恒为 `false` |
| `interpretation` | 指标是描述性变化，也可能由类别比例变化导致 |

诊断不读取真实标签，不给出“必定漂移”结论，不自动修改标准化器。现有融合/路由分数没有经过独立概率校准；`scores` 用于类别排序，`mean_entropy` 用于描述输出分布，不能视为可靠概率置信度。极端有限输入若导致计算溢出会报错。

### `propose(diagnosis, *, training_release=None)`

`diagnosis` 应来自当前活动模型和当前 `revision`，过期时拒绝。始终包含 `{"method":"no_update",...}`。如果声明有匹配本库模式的合法训练标签，再返回固定的小候选集：

| method | parent_id | parameters |
| --- | --- | --- |
| `mlp_continue` | 当前为 MLP / stable 时沿用，否则明确选 stable | `{"epochs":5,"learning_rate":0.001}` |
| `fusion_refit` | 同上 | `{}` |

返回含 `active_model_id`、`revision`、`candidates`、`reason`、`agent_decides=true` 以及 0 次 LLM 调用。此接口不训练、不选择最优模型，也不因漂移数值自动放大搜索。`requires` 是前置条件提示；实际标签、行数及六类覆盖情况在 `train` 校验。

### `train(path, release, *, parent_id=None, method="mlp_continue", parameters=None)`

训练输入只能是合法释放的训练标签，绝不能传入待评估或隐藏验证标签。`parent_id` 默认当前活动模型，训练成功生成新 `a_<id>` 检查点，活动指针不改变。

- `mlp_continue`：父模型必须是 `stable` 或此前生成的 MLP。保留训练标准化参数与 Adam 状态，续训 128→32 tanh→6 softmax 网络。参数仅接受整数 `epochs`（1–10000，默认 5）和 `learning_rate`（0–1，不含 0，默认 0.001）。这不是自动调参接口。
- `fusion_refit`：要求训练数据覆盖六类。使用原第一轮 kinetics 特征，训练集上的中位数/IQR、平衡权重岭回归及固定 0.5 融合比例。参数必须为空。父模型保持不变，保存新的瞬态分支检查点；可指定任一已有父模型。为了避免不必要的层层融合，`propose` 默认选择 MLP 基底。

成功返回 `status="trained"`、新 `model_id`、`parent_id`、`sha256`、`active_model_id`、`training_rows`、`method`、`training_summary`、`resources`。已进入实际拟合后的常见数值/保存失败返回 `status="failed"`、`error`、原活动模型和耗时，并保留此次训练暴露记录；输入/授权/参数错误会抛异常。调用者应先检查 `status`，再读取 `model_id`。

每个新候选保存父版本、父哈希、数据哈希、释放声明与训练参数。v1 提供这两种可新增模型的配方，不提供任意 pickle / 外部可执行检查点导入。

### `evaluate(model_id, path, release)`

同一批合法评估标签上计算候选和当前活动模型的成对指标，不拟合参数。此接口会登记评估暴露及证据，不能当成无副作用的反复查询。

主要返回：

- `evaluation_id`、候选及活动模型的 ID/哈希、评估时的 `revision`；
- `dataset`：文件哈希、释放声明、行数；`evidence`：模式、独立声明、是否重复评估、是否已知开发集及独立证据是否可用；
- `metrics.incumbent` 和 `metrics.candidate`：Macro-F1、Accuracy、每类 precision/recall/F1/support/predicted 和混淆矩阵；
- `eligible`、`reason`、`policy`：本地启用资格与失败检查；
- `agent_summary`：适合 Agent 的简短结论，含 delta Macro-F1 和失败项目；
- `metric_protocol`：主指标 `macro_f1`、次指标 `accuracy`、参与宏平均的类别为真实标签或预测中出现的类别、`policy_is_official_score=false`；
- `resources`：本次两次外层预测及指标计算的耗时。

混淆矩阵行是真实类别、列是预测类别。公开数据分数仍可作为开发诊断，但 `independent_usable=false`，不能通过此结果启用模型。Mock 评估也能在 Mock 版本库中演示资格通过，但 `scope="mock"`；所有评估的 `generalization_verified=false`，避免把单次评估声明当成整体泛化证明。接口不自行聚合最差阶段或估计跨月置信区间；Harness 可按阶段保存指标后统一比较。

### `activate(model_id, evaluation_id, *, expected_revision=None)`

这是总 Harness 明确选择更新后调用的接口，Engine 不自主调用。必须使用本版本库保存的评估；手写指标不能替代证据。检查候选身份、当时的活动模型/版本、合法独立资格、此后训练暴露以及实际模型文件哈希，然后事务性切换活动指针并把原模型压入回滚栈。

成功返回 `status="activated"`、`active_model_id`、递增 `revision` 和 `evidence_scope`。评估不合格则返回 `status="kept"` 和原因，当前模型不变。再次请求同一已启用模型及评估可返回 `already_active`；若传入过期 `expected_revision` 仍先报错。评估过期、模型不匹配或检查点校验失败会抛异常且不切换。

### `rollback(*, expected_revision=None)`

恢复最近一次成功启用前的模型，先校验目标检查点，再原子更新指针、弹出回滚栈并递增 `revision`。成功返回 `status="rolled_back"`。没有历史返回 `status="kept", reason="no_previous_activation"`。多次调用沿成功切换历史逐步恢复；初始 stable 始终保留。回滚不删除候选或数据暴露记录，回滚后的旧评估已过期。

## 4. 本地护栏与正式指标分离

Macro-F1 和 Accuracy 按随包参考评分口径计算。更新策略是研究项目自己的安全护栏，不是官方附加得分：

| 条件 | 默认值 |
| --- | --- |
| Macro-F1 至少提升 | 0.005 |
| 乙醛 F1 | 严格高于当前模型 |
| 乙醛 precision 至少 | 0.65 |
| 其他每个有支持样本的类别 recall 下降上限 | 0.03 |
| 必须有评估支持样本 | 乙醛、乙醇 |
| 证据 | 合法、独立、未重复使用且未训练暴露 |

这套护栏对乙醛有研究目标约束；即使总体改善，但乙醛 F1 未提高，也会保持模型。初始化可指定三个数值阈值：`min_macro_f1_gain`、`min_acetaldehyde_precision`、`max_other_recall_drop`，均须在 [0,1]；其余条件固定。为避免看到评估结果后修改规则，同一库策略不可变。请在独立数据到达前由团队确定策略，不为绕过失败新建库。

## 5. 最小更新调用顺序

下列文件路径、release_id 和独立声明均需 Harness 按真实授权提供。示例的条件判断代表调用方决策，不是 A 模块内部 Agent。

```python
diagnosis = engine.diagnose(incoming_path)
plan = engine.propose(diagnosis, training_release=training_release)
# Harness 的 Agent 选择 no_update 时直接结束；选择训练时：
trained = engine.train(training_path, training_release,
                       parent_id="stable", method="mlp_continue",
                       parameters={"epochs": 5, "learning_rate": 0.001})
if trained["status"] == "trained":
    evidence = engine.evaluate(trained["model_id"], held_out_path, evaluation_release)
    summary_for_agent = evidence["agent_summary"]
    # 仅在 Harness 决定启用后调用（eligible 为必要条件）：
    if harness_wants_update and evidence["eligible"]:
        switched = engine.activate(trained["model_id"], evidence["evaluation_id"],
                                   expected_revision=evidence["revision"])
    # 后续运行需要恢复时，由 Harness 明确请求：
    # engine.rollback(expected_revision=engine.status()["revision"])
```

没有足够独立标签时，可以诊断、提出方案、在合法训练数据上训练，但必须保持活动模型；不要把源内留出、人工扰动压力测试或重复使用的 Batch 2 解释成独立启用依据。

## 6. JSON 逐行适配层

标准输入、标准输出和错误输出使用 UTF-8，Windows 下也相同；调用方需以 UTF-8 发送和接收。

```text
python -m task_a_engine --store ./a_store --mode real
```

请求字段只有 `id`、`action`、`args`；`id` 可省略。`action` 是七个 API 名称或 `status`，`args` 与 Python 参数名相同。示例：

```json
{"id":"p-001","action":"predict","args":{"inputs":"./incoming.jsonl","model_id":"stable"}}
{"id":"t-001","action":"train","args":{"path":"./training.jsonl","release":{"purpose":"train","labels":"released","release_id":"train-01","kind":"real"},"parent_id":"stable","method":"mlp_continue","parameters":{"epochs":5,"learning_rate":0.001}}}
```

成功为 `{"id":...,"ok":true,"result":...}`，可处理的请求错误为 `{"id":...,"ok":false,"error":{"type":...,"message":...}}`，错误后继续读取下一行。进程初始化失败（例如错用版本库模式）会直接退出。空行忽略。适配器不会解释自然语言，也不会将失败操作自动重试。

`id` 仅用于请求对应，**不提供去重或幂等键**。重复 `train` 会生成新版本；重复 `evaluate` 会登记重复暴露并丧失新评估的独立资格。若响应丢失，先检查调用方保存的结果与本地版本记录，避免盲目重试写操作。

SQLite 事务串行化写入；耗时训练也持有写事务。同一个版本库建议由 Harness 排队写操作，连接锁等待为 30 秒。更新和回滚建议始终传 `expected_revision`，遇到冲突重新读取状态，不能静默覆盖其他调用方的切换。每次预测读取一个一致的活动版本快照，期间切换不改变该次预测的模型。

## 7. 资源记录与故障恢复

`resources.cpu_seconds` 是进程 CPU 耗时，`wall_seconds` 是经过时间；`llm_calls=0`、`llm_tokens=0`。`predict` / `diagnose` 的 `prediction_calls=1`，`evaluate=2`，表示 Engine 外层模型预测次数，融合/路由模型内部组件可能执行多次。总 Harness 应自行测量排队、输入解析、模型加载、指纹/哈希和 JSON 传输的端到端开销。

预测计时仅覆盖预测计算；诊断计时包括标准化、预测及指标；评估计时包括两模型预测与评分；训练计时包括拟合及检查点序列化，不含输入加载、输入指纹、父模型加载和最终检查点哈希/登记。不要把这些局部计时当成整阶段端到端延迟。单次极短计算会受系统计时粒度影响，应由 Harness 按批次累计或重复只读推理估计性能。

检查点按新名称写入，不覆盖旧模型。启用/回滚的状态及事件同事务提交，失败时当前指针不变。若文件发布后数据库提交失败，可能留下未注册候选文件，不会自动启用。恢复时先重新打开同一个版本库；不要为恢复方便删除使用记录。检查点损坏时拒绝操作，仍保留稳定模型资产；人工恢复文件应使用原始可信副本并重新校验哈希。

## 8. 执行测试

在解压根目录执行：

```text
python -m unittest discover -s tests -v
python -m examples.mock_harness --output ./demo_run_01
python -m examples.minimal --store ./demo_run_01/engine --mode mock --inputs ./demo_run_01/stage-2-evaluate.jsonl
```

Mock 的人工类别团簇用于确定性覆盖“成功更新”和“拒绝退化候选”分支，并在第三阶段后恢复 stable、重新打开库确认持久化。它不是新模型质量研究，也不能作为真实模式的启用证据。具体实测环境、耗时和通过项目见交付测试报告。
