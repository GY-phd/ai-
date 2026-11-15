# Task A Engine v1

这是供团队总 Harness 调用的 A 任务模块：气体预测、漂移诊断、候选训练、成对评估、本地模型版本切换与回滚。对外提供 Python API 和逐行 JSON 适配器。模块不调用 LLM，不实现 B、C 任务或总调度器；当前接口是项目自定义接口，不是官方比赛 SDK。

## 快速开始

解压后进入本目录，使用 Python 3.10 或以上版本：

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m examples.mock_harness --output ./demo_run_01
```

Mock 输出目录必须尚不存在，以保留每次运行结果。三阶段示例会实际生成新检查点，演示无标签时保持模型、训练与评估后更新、不合格候选拒绝更新、回滚与重新打开版本库。模拟分数只用于检验流程，不是比赛成绩或真实跨月泛化证据。

直接嵌入团队代码：

```python
from task_a_engine import Engine

engine = Engine("./a_model_store")  # 默认 real 模式，首次以 stable 启用
prediction = engine.predict("./incoming.jsonl")
diagnosis = engine.diagnose("./incoming.jsonl")
proposal = engine.propose(diagnosis)
# 将 proposal 交给总 Harness 的 Agent；无合法训练标签时仅返回 no_update。
```

也可安装为普通 Python 包后从其他目录导入：

```text
python -m pip install wheels/task_a_engine-1.0.0-py3-none-any.whl
python -m task_a_engine --store ./a_model_store
```

随包 wheel 已完成离线安装验证；仍需环境中有 NumPy，或允许 pip 安装 requirements 中的 NumPy。若修改源码后重新构建，可用 `python -m pip install .`，构建依赖见 `pyproject.toml`。

测量五个内置模型的计算耗时与 API 耗时：

```text
python -m examples.benchmark --inputs ./incoming.jsonl --store ./benchmark_store_01 --output ./benchmark_01.json
```

基准版本库和输出文件须尚不存在。默认只用输入特征；若文件确实包含合法训练标签，可显式加 `--train-release-id <真实释放记录ID>` 训练两种新候选并记录耗时，仍保持 stable 启用。

进程从标准输入读取一行 JSON，向标准输出返回一行 JSON：

```json
{"id":"s1","action":"status","args":{}}
{"id":"d1","action":"diagnose","args":{"inputs":"./incoming.jsonl"}}
```

所有输入文件路径相对于调用进程的工作目录解析；模型资产相对于安装包定位，无开发机硬编码目录。请为一次比赛运行保留同一个 `store`，供全部 A 任务调用共享数据使用记录。完整参数、返回值和安全边界见 [API 文档](docs/API.md)。可运行的最小读入示例见 [examples/minimal.py](examples/minimal.py)。

## 模型与训练

内置五种已有方案，原检查点继续保留：

| model_id | 作用 |
| --- | --- |
| `stable` | 稳定 NumPy MLP，默认启用及初始回滚基线 |
| `fusion` | 第一轮瞬态特征与 MLP 分数融合 |
| `witness_veto` | 在融合结果上增加乙醛证据否决 |
| `raw_support` | 增加原始瞬态特征支持范围检查与回退 |
| `projected_support` | 增加投影后特征支持范围检查与回退 |

`train` 能实际生成新的 `a_<id>` 候选：`mlp_continue` 续训 MLP 并保留优化器状态；`fusion_refit` 使用合法训练标签重新拟合瞬态分支。训练不会自动替换当前模型，既有候选不会被覆盖。两种训练方法可多次生成不同的新版本；新增架构需要扩展训练与加载实现，v1 不接受任意外部可执行模型。

来源是已有 `A_task_delivery_v1.zip`。本工程复用其中的推理、特征、护栏和模型资产；五轮研究材料及原始结果保持不变。`history/` 中保存原交付 ZIP 与 `research_rounds_01_05.zip`，后者保留五轮代码、模型和报告；运行 Engine 无需解压历史材料，历史实验复现仍需合法 Batch 1 数据。资产哈希和原研究来源哈希保存在 `task_a_engine/vendor/manifest.json`，原始训练及公开开发数据的行指纹用于隔离检查。新模型、评估证据和启用历史存储在调用者指定的版本库中。

## 接入时必须保留的约束

- `predict`、`diagnose` 只使用输入特征；即使文件含 `answer` 也不使用它。
- `train` 与 `evaluate` 要求用途明确的已释放标签声明。已知公开 Batch 2 的标签禁止训练；该数据已参与开发，也不能成为独立启用证据。
- `activate` 只接受同一版本库保存的、绑定候选与当前模型的合格独立评估。样本 ID 和特征指纹用于检测重复与训练泄漏。已有训练暴露的行不能用于评估。
- 同一独立评估集仅首次评估的一个预先选定候选可取得启用资格；随后在该集上比较其他候选属于开发使用。v1 不提供多候选校正或批量盲测接口。
- 真实评估数据以后获得合法训练授权时可以训练；之前的评估证据随之失效。标签授权和跨时间、跨组独立性仍需总 Harness 在来源端保证，布尔声明不构成外部认证。
- `mode="mock"` 的数据与证据只能进入 Mock 版本库，不能启用真实版本库中的模型。不要通过新建空版本库清除数据暴露历史。
- Macro-F1 是评分指标，本地乙醛/乙醇护栏是额外模型更新规则。通过护栏不等于已证明长期泛化。

`evaluate` 的 `agent_summary`、`diagnose` 和 `propose` 适合交给 Agent；无需发送逐样本特征或整个版本库。计算结果提供耗时及调用计数，LLM 调用数和 Token 数始终为 0，由总 Harness 统一记录 SwanLab 与总资源开销。

## 依赖与验证范围

运行时仅依赖 NumPy 和 Python 标准库，测试使用 `unittest`。已使用本地 Windows、Python 3.13.9、NumPy 2.2.6 环境执行验证；具体结果见 [测试报告](reports/TEST_REPORT.md)、[CPU 基准结果](reports/benchmark.json) 和 [三阶段 Mock 结果](reports/mock_results.json)。声明支持范围为 Python >=3.10、NumPy >=1.26.4,<3，但官方 Python 3.10–3.12 / NumPy 1.26.4 组合及其他操作系统尚不能视为已独立测试。模型计算依赖 CPU，不需要 GPU、网络服务或 API 密钥。

Batch 1 乙醛缺少真正跨月留出，历史公开 Batch 2 已多次参与开发。本次工程测试验证可运行性、数据隔离与版本流程，不宣称新的独立气体识别成绩。
