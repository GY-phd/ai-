# Design

## Context

见 `proposal.md` — Why。此处只记录塑造实现方案的当前状态与硬约束。

### 硬约束

| 约束 | 来源 | 对设计的影响 |
|---|---|---|
| 平台以 `contest.cpu-mlp.v1` 冻结格式评分，隐藏集携带**父的特征 schema** | 平台参考加载器 | 任何改动**不得**触及特征 schema、前向实现、检查点格式 |
| 评分请求只携带 `{task, inputs}`，**无 sample id** | 平台预测协议 | 不得引入 id 派生特征列；只能改训练过程（lr / epoch / 分布 / 显式宽度迁移） |
| 镜像**完全离线**，不得做网络探测 | 评测环境 | 模型 ID 解析必须**纯本地**，不得校验端点可用性 |
| 权重文件（`.npz`/`.pkl`/`.pt`/`.safetensors`…）**禁止入库** | 竞赛规则 | 验收基线产物只能落 JSON/Markdown，不得留存 checkpoint |
| 镜像入口须为**唯一**，退出码明确 | 试运行反馈 | View 不得派生常驻进程（否则入口不再是唯一） |
| 每个 profile 10 轮，单轮 `wall_seconds` 600 | 平台帧 | 基线工具必须复用 `offline_host` 的真实数据包，不能另造数据 |

### 当前状态

- `harness/llm.py` 有 `MODEL_ALIASES`（点号→斜杠）与 `resolve_model_id()`，但**只处理短名映射**，无派生 ID 前缀规则。
- `evo/src/evo/llm.py` 的 `_check_model_allowed()` 用**归一化包含匹配**（`norm(m) in norm(model)`），处理派生 ID 与分隔符变体，但规则**与前者不同源**。
- `layers/view/` 七个端点齐备，镜像入口未引用。
- `tools/run_harness_offline.py` 可跑 N 轮闭环，但输出是「每轮明细」，**无跨版本对比口径**。

---

## Goals / Non-Goals

**Goals:**

- 模型 ID 判定收敛为**单一可复用入口**，规则覆盖短名映射、斜杠透传、派生 ID 前缀、分隔符变体、显式豁免优先级。
- 验收基线可在**同父同种子**前提下产出 A/B/C 三 profile 的对比表，并作为"性能更优"的书面证据。
- View 层的部署形态被**显式声明**，消除"死代码"与"线上服务"两种误读。

**Non-Goals（设计级边界）:**

- 不改变 `contest.cpu-mlp.v1` 格式、不改特征 schema、不动前向实现——**任何**设计选项若触碰此项即被否决。
- 不引入 HTTP 服务、不引入常驻进程、不引入新第三方依赖。
- 不删除 `evo/` 目录（它是早期对照实现与 `evo` 侧测试的宿主）。
- 不重写 `offline_host`（它是平台协议的本地替身，已被 SDK 与镜像共同依赖）。

---

## Decisions

### D1 — 模型 ID：权威表 + 归一化包含匹配，统一到 `harness/llm.py`

**选择**：把 `harness/llm.py` 升格为模型身份的唯一真源，新增两个公开函数：

- `resolve_model_id(model_id)` — 形态规范化（点号短名 → 斜杠平台名），保持现有语义。
- `is_allowed_model(model_id, allowed, derived=())` — **合规判定**，把 `evo` 侧的三层判定规则原样搬过来。

`evo/src/evo/llm.py` 删除自有 `norm` lambda 与 `derived_models` 判定分支，改为调用 `harness.llm.is_allowed_model`。

**为什么这样切分**：`resolve_model_id` 解决的是「**拼写**」，`is_allowed_model` 解决的是「**合规**」。两者语义不同、调用点不同（前者用于发请求，后者用于放行与否），但**规则同源**——都基于同一份归一化函数与别名表。分成两个函数而非一个，是因为「把一个非法模型名改写成合法拼写」会掩盖违规。

**归一化规则**（沿用 `evo` 侧已被测试覆盖的语义）：`"".join(ch for ch in s.lower() if ch.isalnum())`——忽略大小写与所有分隔符，因此 `Qwen3.8-27B` / `Qwen3-8-27B` / `qwen3_8_27b` / `Qwen/Qwen3.8-27B` 归一化后**完全相同**。

**判定优先级**（保持 `evo` 现行顺序，不改语义）：

1. **显式豁免**：派生 ID 出现在 `derived` 集合中 → 允许（否则后训练模型名会被字面匹配误拒）。
2. **基座合规**：归一化后**包含**允许集合中任一基座 → 允许。这条同时覆盖派生 ID（`qwen3.8-27b-ft-20261115` 归一化后包含 `qwen38 27b`）与分隔符变体。
3. 否则拒绝。

**为什么不直接让 `evo` 调 `resolve_model_id`**：`resolve_model_id` 只认**精确短名**，返回斜杠形态；而 `evo` 的允许集合来自配置、可能是任意拼写。用精确匹配替换包含匹配会让 `qwen3-8-27b` 这类变体被拒——**收紧了现有行为**，属回归。故采用「归一化包含」而非「精确别名」。

**备选方案否决**：
- *把 `evo` 侧规则整体搬进 `harness`，`evo` 直接 import `resolve_model_id`*——否决：`resolve_model_id` 的精确匹配语义会拒掉现有测试覆盖的变体。
- *删除 `evo/`*——否决：超出本变更范围，且会破坏 `tests/test_core.py` 等一批对照测试。
- *保留双实现加注释「保持同步」*——否决：这正是本次要消除的缺陷本身。

### D2 — 验收基线：`tools/benchmark_profiles.py`，复用 `offline_host` 真实数据

**选择**：新增工具，对每个 profile 执行：

1. **基线测量**：直接加载父检查点，在公共验证集上测主指标（不训练）。这是"未优化"的参照点。
2. **优化后测量**：跑 N 轮闭环（`harness.py train` 经 `offline_host`），取**被采纳的候选**中主指标最优者；若零采纳则退回父检查点。
3. **对比**：产出 before/after、增益方向、采纳率、单位增益。

**关键点**：两次测量使用**同一父检查点、同一种子、同一验证集**——这是规格 `acceptance-benchmark` 里"对照测量的可比性前提"的实现。基线测量**不做任何训练**，因此它就是父检查点在验证集上的分数，天然是同条件的参照。

**为什么不复用 `run_harness_offline.py` 的产物**：该脚本输出的是「每轮明细」（`validation_before`/`validation_after`），第一轮的 `validation_before` 恰好等于父检查点的分数，**在语义上就是基线测量**。所以新工具**包装**它而非重写闭环——避免两套闭环实现漂移。

**产物结构**：

```
runs/benchmark/
  benchmark.json      # 机器可读：per-profile 结构化字段
  benchmark.md        # 人可读：三 profile 对比表
```

**确定性保证**：种子固定为父检查点 `meta["seed"]`（缺省 42），父检查点摘要记录在产物中；同条件下两次运行数值一致（规格要求）。

**为什么不用 `time`-based 随机性**：`_oversample_indices` 用 `np.random.default_rng(seed)`，seed 固定即确定。

### D3 — View 保持进程内库，用**文档 + 包级声明**显式化

**选择**：不在镜像内启动 HTTP 服务。在两处显式声明：

- `contest_baselines/layers/view/__init__.py` 的模块 docstring 顶部加入「部署形态」小节。
- `docs/` 与镜像 README 说明 View 的可达方式（进程内 import 调用）。

**为什么用文档而非代码断言**：代码层面的"启动断言"（如在入口检查某 flag）会在镜像里引入分支，而入口的唯一性是试运行反馈的重点。**加标志位 = 增加入口分支 = 增加被拒风险**。文档声明零运行时成本，且评审可读。

**备选方案否决**：
- *镜像内拉起 server*——已由用户决策否决（多进程管理风险）。
- *删除 `layers/view/`*——否决：它是评审可读性的载体，删掉反而弱化主观分。

### D4 — 规格治理以 OpenSpec 原生工件承载

**选择**：不引入自制的流程检查脚本，直接使用 OpenSpec 的 `proposal → specs → design → tasks` 四件套与 `openspec validate --strict` 作为门禁。

**理由**：重复造轮子的流程校验器没有增量价值，而 OpenSpec 已被安装且原生支持中文、strict 校验与 delta spec。

---

## Risks / Trade-offs

**[Risk] `evo` 侧行为被收紧** → `is_allowed_model` 采用与 `evo` **完全一致**的归一化与包含匹配语义；`tests/test_core.py` 已覆盖 `qwen3.8-27b` / `Qwen3-8-27B` / `qwen3_5_4b` 等变体，改造后必须全绿，任何失败即为回归而非"改进"。

**[Risk] `evo` 依赖 `harness` 包造成导入环** → `harness/llm.py` 只依赖标准库（`json`/`time`/`urllib`），`evo` → `harness` 是单向依赖，无环。

**[Risk] 基线工具因数据路径缺失而整体失败** → `offline_host.package_root()` 已支持 `CONTEST_PACKAGES` 环境变量覆盖；工具 SHALL 支持 `--packages` 参数；单 profile 失败 SHALL NOT 阻断其余 profile（规格已约束）。

**[Risk] 30 轮 × 3 profile 离线演练耗时过长** → 基线工具默认轮数 SHALL 可配（`--rounds`，默认 10）；完整 30 轮作为**验收阶段**单独跑一次并归档产物。

**[Risk] 产物中残留权重** → 基线工具 SHALL **只写 JSON/Markdown**，绝不拷贝 `.npz`；`tools/verify_repo.py` 的零权重检查作为最后一道防线。

**[Trade-off] 「单次精确短名映射」与「归一化包含匹配」并存** → 两套映射语义不同但**规则同源**（共享归一化函数与别名表），且各自职责清晰（拼写 vs 合规）。代价是需要读者理解两个函数；换来的是 `evo` 既有行为零回归。

**[Trade-off] View 层保持不可达** → 评审若期望线上 API 会认为功能"未完成"；用 `docs/` 显式声明 + 保留可运行库 + 测试证明端点契约正确来对冲。

---

## Migration Plan

1. **阶段一 · 规格**：产出 `proposal.md` / `specs/`（4 个能力）/ `design.md` / `tasks.md`，`openspec validate --strict` 通过。
2. **阶段二 · 代码**：按 `tasks.md` 顺序改码，每项改完即跑对应测试。
3. **阶段三 · 验收**：`pytest tests -q` 全绿 → 跑三 profile 基线对比 → 产出 `runs/benchmark/`。
4. **阶段四 · 发布**：构建干净仓库 → `tools/verify_repo.py` 零权重校验 → 推送 `origin/main`。
5. **回滚策略**：本变更均为增量改动，回滚 = `git revert` 对应提交；无数据迁移、无外部契约破坏（`evo` 的对外行为保持不变）。

---

## Open Questions

无。规格已覆盖全部行为边界，设计已锁定所有影响实现路径的选项。
