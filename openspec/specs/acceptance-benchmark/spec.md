# acceptance-benchmark Specification

## Purpose

本能力规定「性能更优」这一验收主张的**量化口径**：系统必须能在同一父检查点、同一随机种子、同一公共验证集下，分别测出「未优化基线」与「优化后」的主指标，并给出可复现、可审计的对比表与结论。

## Requirements

### Requirement: 对照测量的可比性前提

系统 SHALL 在两次测量之间保持以下条件完全一致：父检查点（含其摘要）、随机种子、公共验证集投影、模型前向实现。任一条件不一致时，工具 SHALL 拒绝出具对比结论。

#### Scenario: 同条件测量产生有效对比

- **GIVEN** 同一父检查点、同一种子、同一验证集
- **WHEN** 分别执行基线测量与优化后测量
- **THEN** 两次测量 SHALL 产生可比的主指标数值
- **AND** 工具 SHALL 输出对比结论

#### Scenario: 条件不一致时拒绝出结论

- **GIVEN** 两次测量的父检查点摘要不同
- **WHEN** 工具尝试生成对比结论
- **THEN** 工具 SHALL 拒绝输出性能结论
- **AND** SHALL 明确指出不一致的条件

### Requirement: 三 profile 覆盖

验收基线 SHALL 覆盖数据集 profile A、B、C 三个 profile，每个 profile 独立出具对比结论。任一 profile 的测量失败 SHALL NOT 阻断其余 profile 的结论产出，但 SHALL 在产物中标记该 profile 为失败并记录原因。

#### Scenario: 三 profile 全部成功

- **GIVEN** A、B、C 三 profile 的数据包齐备
- **WHEN** 执行验收基线
- **THEN** 产物 SHALL 包含三个 profile 的对比结论

#### Scenario: 单 profile 失败不影响其余

- **GIVEN** profile B 的数据包缺失
- **WHEN** 执行验收基线
- **THEN** profile A 与 C SHALL 仍产出对比结论
- **AND** profile B SHALL 被标记为失败并附失败原因

### Requirement: 对比表结构

每个 profile 的对比结论 SHALL 包含：profile 键、任务类型、主指标名称、父检查点摘要、种子、基线主指标值、优化后主指标值、增益方向与幅度、采纳的轮次数与总轮次数。增益方向 SHALL 遵循任务语义——分类任务越高越好，回归任务越低越好。

#### Scenario: 分类 profile 的增益方向

- **GIVEN** 一个分类 profile，优化后 Macro-F1 高于基线
- **WHEN** 生成对比表
- **THEN** 增益 SHALL 被标记为正向改善
- **AND** SHALL 报告 Macro-F1 的基线值与优化后值

#### Scenario: 回归 profile 的增益方向

- **GIVEN** 一个回归 profile，优化后 RMSE 低于基线
- **WHEN** 生成对比表
- **THEN** 增益 SHALL 被标记为正向改善
- **AND** SHALL 报告 RMSE 的基线值与优化后值

### Requirement: 采纳率与单位增益

每个 profile 的对比结论 SHALL 报告**采纳率**（采纳轮次 ÷ 总轮次）与**单位增益**（主指标改善量 ÷ 采纳轮次）。当采纳轮次为零时，单位增益 SHALL 记为不可用而非零，以免把"无动作"误读为"无效果"。

#### Scenario: 采纳率与单位增益被计算

- **GIVEN** 一个 profile 共 10 轮，其中 3 轮被采纳
- **WHEN** 生成对比表
- **THEN** 采纳率 SHALL 为 0.3
- **AND** 单位增益 SHALL 为主指标改善量除以 3

#### Scenario: 零采纳时单位增益不可用

- **GIVEN** 一个 profile 全部轮次均未被采纳
- **WHEN** 生成对比表
- **THEN** 单位增益 SHALL 标记为不可用
- **AND** SHALL NOT 报告为 0

### Requirement: 产物格式与可复现性

验收基线 SHALL 同时产出 JSON 与 Markdown 两种格式，SHALL 记录运行时使用的父检查点摘要与种子，使第三方可在同条件下复现。产物 SHALL 为确定性输出：同条件下两次运行的数值 SHALL 一致。

#### Scenario: 双格式产物

- **WHEN** 执行验收基线
- **THEN** 输出目录 SHALL 同时包含 JSON 产物与 Markdown 产物
- **AND** 两者 SHALL 表达同一组结论

#### Scenario: 产物携带复现信息

- **WHEN** 检查任一产物的内容
- **THEN** 其中 SHALL 记录父检查点摘要与随机种子
- **AND** SHALL 记录每个 profile 的轮次数与采纳数

#### Scenario: 同条件重复运行结果一致

- **GIVEN** 相同的数据包、父检查点与种子
- **WHEN** 连续两次执行验收基线
- **THEN** 两次产物的主指标数值 SHALL 完全一致
