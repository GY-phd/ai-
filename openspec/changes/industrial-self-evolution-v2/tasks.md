# Tasks

## 1. 模型身份统一（model-identity-resolution）

- [x] 1.1 在 `harness/llm.py` 新增 `normalize_model_id()`（忽略大小写与所有分隔符的归一化，并去掉厂商前缀）与 `is_allowed_model(model_id, allowed, derived=())`（三层判定：显式豁免 → 基座包含 → 拒绝），并在模块 docstring 中声明本模块为模型身份的唯一真源。**验证**：`python -c "from harness.llm import is_allowed_model; assert is_allowed_model('qwen3.8-27b-ft-20261115', ['Qwen3.8-27B'])"` 通过。
- [x] 1.2 扩展 `MODEL_ALIASES` 覆盖 4B 与 instruct 形态，并补充 `resolve_model_id` 对空输入与未知 ID 的行为（空串返回空串，未知原样透传）。**验证**：`pytest tests/test_config.py -q` 全绿（实测 76 passed，与 test_core 一并通过）。
- [x] 1.3 修改 `evo/src/evo/llm.py`：删除自有 `norm` lambda 与 `derived_models` 内联判定分支，改调 `harness.llm.is_allowed_model`，并保留 `LLMClient.model_allowed` 的对外属性名不变。**验证**：`pytest tests/test_core.py -q` 全绿（该文件已覆盖变体、派生模型、非千问拒绝三条路径）。
- [x] 1.4 新增 `tests/test_model_identity.py`，覆盖：单一真源扫描（仓库内除权威表外无第二套变体正则）、点号短名映射、大小写不敏感、斜杠透传、未知 ID 透传、派生 ID 前缀匹配、显式派生优先、空输入安全。**验证**：`pytest tests/test_model_identity.py -q` 全绿（实测 37 passed）。
- [x] 1.5 在 `docs/` 中补充「模型身份解析」一节，写明唯一真源位置、两种拼写的对应关系、派生 ID 规则。**验证**：人工核对文档与 `harness/llm.py` 的实现一致（`docs/HARNESS.md` §3.1）。

## 2. 验收基线工具（acceptance-benchmark）

- [x] 2.1 新增 `tools/benchmark_profiles.py` 的骨架：参数解析（`--datasets`、`--rounds`、`--packages`、`--out`），逐 profile 循环，单 profile 异常捕获后标记失败并继续其余 profile。**验证**：`python tools/benchmark_profiles.py --help` 正常输出且列出全部参数。
- [x] 2.2 实现基线测量：经 `contest_baselines.offline_host.package_root()` 定位真实数据包，加载父检查点，在公共验证集上测主指标（不训练）。**验证**：对 profile A 输出一个非空的主指标数值（实测 0.7140）。
- [x] 2.3 实现优化后测量：复用 `tools/run_harness_offline.py` 的 `_HistoryHost` / `_run_round` / `_ledger_entry`，跑 N 轮闭环，取被采纳候选中的最优者；零采纳时退回父检查点。**验证**：报告中出现「优化后」数值且与被采纳轮次一致（实测 A=0.7192 / B=456.4174 / C=50.6834）。
- [x] 2.4 实现对比与统计：增益方向（分类越高越好 / 回归越低越好）、采纳率、单位增益（零采纳时标为不可用而非 0），并在产物中写入父检查点摘要与种子。**验证**：`benchmark.json` 含 `parent_checkpoint_sha256`、`seed`、`adoption_rate`、`unit_gain` 字段。
- [x] 2.5 实现双格式产物输出：`benchmark.json` 与 `benchmark.md`，后者为三 profile 对比表。**验证**：`runs/benchmark/` 下两文件同时存在且表达同一组结论。
- [x] 2.6 新增 `tests/test_benchmark.py`，覆盖：方向判定（分类/回归各一）、采纳率计算、单位增益在零采纳时不可用、产物结构字段齐全、同条件两次运行数值一致。**验证**：`pytest tests/test_benchmark.py -q` 全绿（实测 19 passed）。
- [x] 2.7 确认基线工具**不写入任何权重文件**：全流程只写 `.json` 与 `.md`。**验证**：探针运行后 `runs/_probe` 内只有 `benchmark.json` 与 `benchmark.md`，无任何权重后缀文件。

## 3. View 部署形态显式化（view-layer-contract）

- [x] 3.1 在 `contest_baselines/layers/view/__init__.py` 的模块 docstring 中加入「How the View is deployed」小节：声明 View 为进程内只读库、不随镜像自动启动 HTTP 服务、其在镜像中的可达方式。**验证**：人工核对 docstring 内容与规格 `view-layer-contract` 的要求一致。
- [x] 3.2 在 `docs/` 中补充 View 层的可达方式说明（进程内 import 调用的示例代码），并明确镜像入口不派生常驻服务。**验证**：文档含可复制的调用示例（`docs/系统运行说明.md` §4.3.1）。
- [x] 3.3 新增 `tests/test_view_deployment.py`，覆盖：镜像入口不启动 View 服务（断言入口模块中无 View 服务启动调用）、View 在进程内可直接调用并返回带 `kind`/`version` 的答案、查询无写副作用、七个端点齐备。**验证**：`pytest tests/test_view_deployment.py -q` 全绿（实测 11 passed）。

## 4. 集成验收

- [x] 4.1 跑全量测试 `pytest tests -q`，记录通过数（基线为 252 passed, 1 skipped），确认无回归。**验证**：全部测试通过且通过数不低于基线。**实测**：319 passed, 1 skipped, 102 subtests passed（较基线 +67）。
- [x] 4.2 执行 `python tools/benchmark_profiles.py --datasets A,B,C --rounds 10 --out runs/benchmark`，产出 A/B/C 三 profile 的对比表。**验证**：`runs/benchmark/benchmark.json` 与 `benchmark.md` 存在，三个 profile 均有对比结论。**实测**：A 0.7140→0.7192、B 681.1598→456.4174、C 52.8686→50.6834，三者均为 improved。
- [x] 4.3 跑完整 30 轮离线演练 `python tools/run_harness_offline.py --rounds 10 --datasets A,B,C --out runs/harness-offline`，归档每轮明细作为性能证据。**验证**：`runs/harness-offline/run_report.json` 含三 profile 各 10 轮记录。**实测**：30 轮全部完成，采纳 20 轮（A 1 / B 9 / C 6）。
- [x] 4.4 跑 `python tools/verify_repo.py --working-tree`，确认零权重违规。**验证**：输出中权重检查项全部通过。**实测**：`OK 无权重文件 / 6635 个文件`。
- [x] 4.5 核对 `MANIFEST.json` 与 `GIT_PUSH.md` 的未提交改动，确保与本变更一致后一并落库。**验证**：`git diff` 确认改动内容属预期。**实测**：`GIT_PUSH.md` 为「仓库根目录即工作仓库 + 三道权重防线」的说明补全；`MANIFEST.json` 已用 `tools/refresh_manifest.py` 重新生成（321 文件 / 3465928 字节 / weights=False），并把 `openspec/` 纳入清单。

## 5. 发布

- [ ] 5.1 构建干净发布仓库并核对清单。**验证**：`python tools/build_github_repo.py` 成功且产物体积在预算内。
- [ ] 5.2 提交并推送到 `origin/main`。**验证**：`git push` 成功且 `git log origin/main -1` 与本地 HEAD 一致。
- [ ] 5.3 归档 OpenSpec 变更：`openspec archive industrial-self-evolution-v2`，规格并入 `openspec/specs/`。**验证**：`openspec list --specs` 列出四个能力。
