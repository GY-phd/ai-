# Proposal

## Why

当前系统（`contest_baselines/` + `harness/`）已通过 252 项单元测试，架构清晰、闭环完整，但**缺少一份可追溯的规格基线**：需求、设计、代码、验收四者之间没有书面契约，导致「改了哪条需求、满足哪条验收、由哪个测试证明」全靠口头约定。同时存在三处具体缺陷：

1. **模型 ID 白名单双源**：`harness/llm.py` 的 `MODEL_ALIASES` 与 `evo/src/evo/llm.py` 的 `qwen3827b` 变体正则使用不同拼写规则，同一模型名在不同模块可能被判定为不同结果。
2. **验收缺量化基线**：`tools/run_harness_offline.py` 能跑 30 轮闭环，但没有「优化前 vs 优化后」的跨 profile 对比口径，无法证明"性能更优"。
3. **镜像内 View 层为不可达代码**：`layers/view/` 7 个端点实现完整，但镜像入口从不启动 HTTP 服务（已决策：保持为库，不在镜像内拉起）。

本变更按 **需求分析 → 详细设计 → 代码开发 → 验收** 四阶段推进，先以 OpenSpec 固化规格，再据规格改代码，最后以测试与离线演练对比表证明改进并推送 GitHub。

## What Changes

- **新增规格基线**：以 OpenSpec `spec-driven` schema 产出 `proposal → specs → design → tasks` 四件套，作为后续所有改动的书面契约。
- **统一模型 ID 解析（BREAKING for `evo`）**：`harness/llm.py` 的 `MODEL_ALIASES` 成为**唯一权威表**；`evo/src/evo/llm.py` 移除自有变体正则，改为复用 `harness.llm.resolve_model_id`。`evo/` 保留为早期对照实现，不删除目录。
- **新增验收基线工具**：`tools/benchmark_profiles.py`，对 A/B/C 三 profile 分别执行「同父检查点 + 同种子」的基线测量与优化后测量，产出结构化对比表（主指标 before/after、采纳率、单位增益），JSON 与 Markdown 双输出。
- **View 层定位显式化**：`layers/view/` 保持纯库形态，在 `layers/view/__init__.py` 与镜像文档中显式声明"不随镜像自动启动"，避免评审误认为其是死代码或误认为线上服务。
- **补齐缺失的回归测试**：为模型 ID 统一、验收基线工具、View 层"只读不启动"契约各补测试。
- **MANIFEST / GIT_PUSH 同步**：`MANIFEST.json` 与 `GIT_PUSH.md` 的未提交改动随本变更一并落库。

**Non-goals（明确排除）**：
- 不在镜像内启动 HTTP 服务（已决策 `keep-as-lib`）。
- 不删除 `evo/` 目录（已决策 `keep-both` 的变体：统一白名单但保留实现）。
- 不改变 `contest.cpu-mlp.v1` 参考格式、不改变特征 schema、不引入新第三方运行时依赖。

## Capabilities

### New Capabilities
- `model-identity-resolution`: 模型 ID 的**单一权威解析规则**。规定 `harness/llm.py` 的别名表是唯一真源，`resolve_model_id` 是唯一入口，全仓库不得存在第二套变体正则；规定点号短名与斜杠平台名的双向映射、未知 ID 的透传行为、以及 PyTRiO 后训练派生 ID 的前缀匹配规则。
- `acceptance-benchmark`: **验收基线的量化口径**。规定「性能更优」如何被证明：同父检查点、同种子、同公共验证集下，基线测量与优化后测量的对比方式；规定 A/B/C 三 profile 的对比表结构、采纳率定义、单位增益定义，以及 JSON/Markdown 双格式产物。
- `spec-lifecycle-governance`: **规格驱动的变更治理**。规定需求分析→详细设计→代码开发→验收四阶段各自的准入准出条件；规定每条需求必须可由至少一条测试证明；规定「无规格不得改码」的例外通道（纯重构/工具/文档）。

### Modified Capabilities
- `view-layer-contract`: View 层的**部署形态契约**发生变化——由"实现完整但未声明部署方式"改为"显式声明为进程内只读库，不随镜像自动启动 HTTP 服务"。API 端点集合与只读性不变。

## Impact

**代码**
- `harness/llm.py` — `MODEL_ALIASES` 升格为唯一权威表（新增派生 ID 前缀规则）。
- `evo/src/evo/llm.py` — 删除自有变体正则，改调 `harness.llm.resolve_model_id`。
- `tools/benchmark_profiles.py` — **新增**验收基线工具。
- `contest_baselines/layers/view/__init__.py` — 显式部署形态声明。
- `tests/` — 新增/修改测试文件（模型 ID、基线工具、View 契约）。

**产物**
- `openspec/specs/<capability-path>/spec.md` × 3 新增，1 个 delta。
- `runs/benchmark/` — 验收对比表（JSON + Markdown）。

**依赖**：无新增第三方依赖（全部基于 numpy + 标准库）。

**风险**
- `evo/` 改动触及早期对照实现，其测试需同步通过；若 `evo` 侧存在依赖旧正则的行为，将一并调整并记录。
- 验收基线工具需在 600s wall-clock 内跑完三 profile 的对照测量，需复用 `offline_host` 的真实数据包而非构造数据。
