# model-identity-resolution Specification

## Purpose

本能力规定模型 ID 在整个仓库中的**唯一解析规则**：平台下拉框里的斜杠 ID（如 `Qwen/Qwen3.8-27B`）与赛题文本里的点号短名（如 `Qwen3.8-27B`）必须被解析为**同一个**模型，且这一映射只允许存在一份权威表，任何模块不得私自定义变体正则。

## Requirements

### Requirement: 单一权威别名表

系统 SHALL 提供**唯一一处**模型 ID 别名表作为唯一真源，并 SHALL 保证全仓库不存在第二套等价的变体匹配规则。任何模块需要判断"这个 ID 是哪个模型"时 MUST 复用该表提供的解析入口。

#### Scenario: 仓库内无重复变体规则

- **GIVEN** 仓库包含模型 ID 相关代码
- **WHEN** 扫描所有用于判定模型身份的变体匹配规则
- **THEN** 恰好存在一处别名表与解析入口
- **AND** 其余模块均调用该入口而无自有等价正则

#### Scenario: 早期对照实现复用同一入口

- **GIVEN** 早期对照实现目录中存在模型 ID 判定逻辑
- **WHEN** 该逻辑判定一个模型 ID 是否在允许集合内
- **THEN** 它 SHALL 通过统一解析入口得出结论
- **AND** 其判定结果 SHALL 与权威表一致

### Requirement: 点号短名映射到斜杠平台名

系统 SHALL 将点号短名映射为平台下拉框使用的斜杠 ID，映射 SHALL **大小写不敏感**。已在斜杠形态的 ID SHALL 原样返回。

#### Scenario: 短名映射

- **GIVEN** 配置中写入点号短名 `Qwen3.8-27B`
- **WHEN** 系统解析该 ID
- **THEN** 返回斜杠形态 `Qwen/Qwen3.8-27B`

#### Scenario: 小写短名映射

- **GIVEN** 配置中写入小写短名 `qwen3.8-27b`
- **WHEN** 系统解析该 ID
- **THEN** 返回斜杠形态 `Qwen/Qwen3.8-27B`

#### Scenario: 斜杠 ID 原样返回

- **GIVEN** 输入 ID 已包含斜杠 `Qwen/Qwen3.8-27B`
- **WHEN** 系统解析该 ID
- **THEN** 返回值与输入完全相同

### Requirement: 未知 ID 透传

系统 SHALL 将未知 ID 原样返回，SHALL NOT 抛错、SHALL NOT 猜测或改写。是否允许使用该模型由独立的允许集合校验决定，解析器不负责该判定。

#### Scenario: 未知 ID 不被改写

- **GIVEN** 输入 ID 不在别名表中
- **WHEN** 系统解析该 ID
- **THEN** 返回值与输入完全相同
- **AND** 解析过程 SHALL NOT 抛出异常

### Requirement: 后训练派生 ID 前缀匹配

系统 SHALL 支持 PyTRiO 后训练派生模型 ID（形如 `<基座ID>-ft-<日期后缀>`）。当允许集合只含基座 ID 时，系统 SHALL 认可其派生 ID；若允许集合中已显式列出派生 ID，则以显式项为准。

#### Scenario: 派生 ID 被基座允许集合覆盖

- **GIVEN** 允许集合为 `["Qwen/Qwen3.8-27B"]`
- **WHEN** 判定 ID `qwen3.8-27b-ft-20261115` 是否允许
- **THEN** 判定结果为允许

#### Scenario: 显式列出的派生 ID 优先

- **GIVEN** 允许集合同时包含基座 ID 与某个显式派生 ID
- **WHEN** 判定该显式派生 ID
- **THEN** 判定结果为允许
- **AND** 该判定 SHALL NOT 因前缀规则而被改写

#### Scenario: 短名派生 ID 同样被覆盖

- **GIVEN** 允许集合为 `["Qwen3.8-27B"]`
- **WHEN** 判定短名形态的派生 ID `qwen3.8-27b-ft-20261115`
- **THEN** 判定结果为允许

### Requirement: 空输入处理

系统 SHALL 将空白或空字符串 ID 处理为空字符串返回，SHALL NOT 抛出异常。

#### Scenario: 空字符串安全返回

- **GIVEN** 输入为 `None` 或空字符串
- **WHEN** 系统解析该 ID
- **THEN** 返回空字符串
- **AND** 解析过程 SHALL NOT 抛出异常
