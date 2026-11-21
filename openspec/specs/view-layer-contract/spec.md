# view-layer-contract Specification

## Purpose

本能力规定只读 View 层（账本、轮次、度量、champion、假设、预测查询）的对外契约，以及它的**部署形态**：View 是进程内只读库，不随竞赛镜像自动启动 HTTP 服务。

## Requirements

### Requirement: 部署形态为进程内只读库

View 层 SHALL 作为进程内只读库提供查询能力，SHALL NOT 随竞赛镜像自动启动 HTTP 服务进程。镜像入口 SHALL NOT 派生任何常驻服务线程或子进程来提供 View 查询。

#### Scenario: 镜像运行期间无常驻服务

- **GIVEN** 镜像以正式入口启动并完成一轮闭环
- **WHEN** 检查运行期间的进程树
- **THEN** SHALL NOT 存在为提供 View 查询而派生的常驻进程
- **AND** 主入口 SHALL 在轮次结束后正常退出

#### Scenario: View 仍可在进程内直接使用

- **GIVEN** 一个指向某轮状态目录的 View 实例
- **WHEN** 在进程内调用其查询端点
- **THEN** SHALL 返回该端点契约规定的答案结构
- **AND** SHALL NOT 需要启动任何服务进程

#### Scenario: 部署形态被显式声明

- **GIVEN** 查看 View 层的公共文档或包说明
- **WHEN** 阅读其部署相关说明
- **THEN** SHALL 明确写明 View 为进程内只读库且不随镜像自动启动
- **AND** SHALL 说明其在镜像中的可达方式

### Requirement: 只读性

View 的所有查询 SHALL 为对 Model 层表的纯读取，SHALL NOT 写入任何文件，SHALL NOT 修改任何状态。同一冻结目录上的重复查询 SHALL 返回相同结果。

#### Scenario: 查询不产生副作用

- **GIVEN** 一个指向某轮状态目录的 View 实例
- **WHEN** 依次调用其全部查询端点
- **THEN** 该目录下的文件内容与修改时间 SHALL 保持不变

#### Scenario: 冻结目录上的重复查询一致

- **GIVEN** 一个只读的冻结归档目录
- **WHEN** 连续两次查询同一端点
- **THEN** 两次返回的答案 SHALL 完全一致

### Requirement: 答案结构与版本标记

每个查询端点 SHALL 返回带 `kind` 与 `version` 字段的 JSON 对象，SHALL 可直接序列化为 JSON 而无需额外处理。端点集合 SHALL 至少覆盖：概览、账本、轮次、度量、champion、假设、单样本预测。

#### Scenario: 答案携带种类与版本

- **GIVEN** 任意查询端点被调用
- **WHEN** 检查其返回值
- **THEN** 返回值 SHALL 含 `kind` 字段标识答案种类
- **AND** SHALL 含 `version` 字段标识答案结构版本

#### Scenario: 端点集合完整

- **GIVEN** View 实例
- **WHEN** 枚举其公开查询方法
- **THEN** SHALL 至少包含概览、账本、轮次、度量、champion、假设与单样本预测七类查询

### Requirement: 与协议解耦

View SHALL NOT 依赖竞赛 SDK 的协议帧，SHALL NOT 读写标准输入输出，SHALL NOT 感知当前数据集 profile 的具体含义。

#### Scenario: 无协议依赖

- **GIVEN** View 层的全部源码
- **WHEN** 检查其导入与协议相关引用
- **THEN** SHALL NOT 引入竞赛 SDK 的协议模块
- **AND** SHALL NOT 操作标准输入或标准输出

#### Scenario: 不感知 profile 语义

- **GIVEN** 一个 View 实例读取 A、B、C 三个 profile 的状态目录
- **WHEN** 查询其端点
- **THEN** 返回内容 SHALL 仅反映表中的记录
- **AND** SHALL NOT 因 profile 不同而走不同分支
