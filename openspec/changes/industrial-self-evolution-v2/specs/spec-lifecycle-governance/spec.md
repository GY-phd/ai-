# spec-lifecycle-governance

## Purpose

本能力规定本项目变更的**流程治理规则**：任何改动都必须经过需求分析、详细设计、代码开发、验收四个阶段，每条需求都必须可由测试证明，并明确「无规格不得改码」的例外通道与违规处置。

## ADDED Requirements

### Requirement: 四阶段准入准出

变更 SHALL 依次经过需求分析、详细设计、代码开发、验收四个阶段。每一阶段 SHALL 有明确的准出条件，未满足 SHALL NOT 推进到下一阶段。

#### Scenario: 需求分析阶段的准出

- **GIVEN** 一个新变更被提出
- **WHEN** 需求分析阶段结束
- **THEN** 提案文档 SHALL 已写明变更动因、改动清单、能力边界与影响范围
- **AND** 能力清单 SHALL 非空或显式声明跳过规格

#### Scenario: 详细设计阶段的准出

- **GIVEN** 需求分析已通过
- **WHEN** 详细设计阶段结束
- **THEN** 设计文档 SHALL 已给出数据模式、模块职责、接口契约与状态流转
- **AND** 每个新增能力 SHALL 有对应的规格文件

#### Scenario: 代码开发阶段的准出

- **GIVEN** 详细设计已通过
- **WHEN** 代码开发阶段结束
- **THEN** 每条新增或修改的需求 SHALL 至少有一条测试覆盖
- **AND** 全部测试 SHALL 通过

#### Scenario: 验收阶段的准出

- **GIVEN** 代码开发已通过
- **WHEN** 验收阶段结束
- **THEN** 验收基线 SHALL 已产出可复现的对比结论
- **AND** 提交产物的清单与摘要 SHALL 已核对

### Requirement: 需求可证明性

每条 SHALL/MUST 级需求 SHALL 至少对应一条可自动执行的测试，或一条显式记录为「人工验收」并写明判定方式的检查。需求描述 SHALL 控制在 500 字符以内以保持可读性。

#### Scenario: 每条需求都有对应测试

- **GIVEN** 一份已完成的需求规格
- **WHEN** 核对每条需求与测试的对应关系
- **THEN** 每条需求 SHALL 至少关联一条测试
- **AND** 不存在无任何测试或人工检查对应的需求

#### Scenario: 场景使用可执行格式

- **GIVEN** 一条需求包含若干场景
- **WHEN** 检查场景格式
- **THEN** 每个场景 SHALL 以触发条件开头
- **AND** SHALL 包含至少一条可判定的结果断言

### Requirement: 无规格不得改码的例外通道

纯重构、工具脚本与文档改动可不产出规格文件，但 SHALL 在变更元数据中显式标记跳过规格，且 SHALL 说明理由。仅当改动不改变任何对外可见行为时 SHALL 允许此通道。

#### Scenario: 显式跳过规格

- **GIVEN** 一个仅调整内部实现、不改变对外行为的改动
- **WHEN** 提交该变更
- **THEN** 变更元数据 SHALL 标记跳过规格
- **AND** SHALL 说明其为纯重构的理由

#### Scenario: 改变行为的改动不得跳过

- **GIVEN** 一个改变了对外可见行为的改动
- **WHEN** 提交该变更
- **THEN** 校验 SHALL 拒绝该变更
- **AND** SHALL 要求补齐规格文件

### Requirement: 规格产物的语言与结构约定

规格文档 SHALL 以中文撰写，SHALL 保留 OpenSpec 的结构性标题与 SHALL/MUST 关键字的英文原文。需求标题 SHALL 以行为命名而非以实现命名。

#### Scenario: 语言与关键字约定

- **GIVEN** 一份规格文档
- **WHEN** 检查其正文与关键字
- **THEN** 正文 SHALL 为中文
- **AND** SHALL/MUST SHALL 以英文原文出现

#### Scenario: 需求标题描述行为

- **GIVEN** 一条需求
- **WHEN** 检查其标题
- **THEN** 标题 SHALL 描述系统行为
- **AND** SHALL NOT 仅描述实现方式
