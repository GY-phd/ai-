# 系统运行说明文档

> 赛题提交资料之一。面向**评测机**：说明如何启动、在哪里取结果、失败时怎么办。
> 设计原理见 [HARNESS.md](HARNESS.md)，方法与消融见 [SOLUTION.md](SOLUTION.md)。

---

## 1. 交付物清单

| 交付物 | 位置 | 说明 |
|---|---|---|
| Harness 系统镜像 | 仓库根 `Dockerfile` | `docker build -t zhixun-engine-b .` |
| 模型配置 | `harness/config/harness.json` | 闭环与训练的全部超参 |
| 完整实验记录 | `evidence/` | 账本、指标流、四段式证据链、运行报告 |
| 系统运行说明 | 本文件 | — |
| 代码 | `harness/` `rulkit/` `tools/` `research/` `tests/` | 无权重 |

---

## 2. 启动

### 2.1 镜像方式（评测推荐）

```bash
docker build -t zhixun-engine-b .

docker run --rm --network none \
  -v /path/to/official-data:/data:ro \
  -v /path/to/output:/out \
  zhixun-engine-b
```

`--network none` 表示**完全断网**。这是必须能跑通的形态：
大模型端点不可达时，`harness/llm.py` 的 `available()` 返回 False，
`plan()` 抛 `LLMUnavailable`，`harness/policy.py` 的确定性策略接管，
动作空间与 schema 完全相同，决策记录里标 `fallback: true`。

### 2.2 直接运行（开发机）

```bash
pip install -r requirements.txt
python -m harness.run --workspace runs/harness-b --batch-size 200 --max-experiments 6
```

Windows：`.\run_harness.ps1`；Linux/macOS：`./run_harness.sh`。

### 2.3 环境变量

镜像入口**不接受命令行参数**，配置全部来自环境变量：

| 变量 | 默认 | 含义 |
|---|---|---|
| `RUL_DATA_ROOT` | `/data` | 官方数据包根目录（内含 `public/train.jsonl`、`public/validation.jsonl`、`checkpoint.npz`） |
| `HARNESS_WORKSPACE` | `/out/harness` | 产物输出目录 |
| `HARNESS_BATCH_SIZE` | `200` | 流式批次大小 |
| `HARNESS_MAX_EXPERIMENTS` | `6` | 实验次数硬预算 |
| `HARNESS_MAX_WALL_SECONDS` | `2400` | 墙钟预算（秒） |
| `HARNESS_MAX_TRAIN_SECONDS` | `900` | 训练秒数硬预算 |
| `HARNESS_MAX_TOKENS` | `20000` | 大模型 token 预算 |
| `HARNESS_LLM_MODE` | `auto` | `auto` / `qwen` / `policy` |
| `HARNESS_LLM_BASE_URL` | 空 | OpenAI 兼容端点；空则用策略兜底 |
| `HARNESS_LLM_API_KEY` | 空 | 可留空 |

命令行参数**优先于**环境变量（命令行更具体）。

---

## 3. 输入

```
/data/
├── checkpoint.npz           # 官方静态基线（启动 champion）
└── public/
    ├── train.jsonl          # 训练集
    └── validation.jsonl     # 公开验证集；同时作为漂移流的来源
```

路径在 `harness/config.py::resolve_data` 解析：凡以 `data/` 开头的配置项，
都会被重定位到 `RUL_DATA_ROOT` 之下。

---

## 4. 输出

全部落在 `HARNESS_WORKSPACE`（镜像里是 `/out/harness`）：

| 文件 | 内容 |
|---|---|
| `run_report.json` | **主结果**：基线/最终指标、预算消耗、逐周期决策、champion 版本链、四段式证据链汇总 |
| `ledger.jsonl` | 决策账本，**哈希链**：每条含前一条的 SHA-256 |
| `hypothesis_chain.jsonl` | 四段式证据链（读历史→提假设→做实验→做决策），**含"什么都不做"的周期** |
| `swanlab.jsonl` | SwanLab 形状的指标流 |
| `registry/champion.json` | 当前 champion 指针 + 指标 |
| `registry/champions.jsonl` | champion 版本链 |
| `experiments/` | 每个候选的检查点与逐 trial 指标 |

校验：

```bash
python tools/verify_ledger.py --ledger /out/harness/ledger.jsonl
# OK  9 records, 9 decisions, 2 promotions, chain intact
```

---

## 5. 运行行为

### 5.1 无人干预

`harness/orchestrator.py::Harness.run()` 是单进程循环：
逐批次读流 → 漂移监测 → 诊断归因 → 预算内实验 → 门控 → 必要时回滚。
**没有 `input()`、没有人工确认、没有交互式提示。**

### 5.2 受控自改进（防无限迭代）

| 约束 | 实现 |
|---|---|
| 实验次数 | `budget.max_experiments`，超限即降级 `do_nothing` |
| 训练时长 | `budget.max_train_seconds` |
| 墙钟时长 | `budget.max_wall_seconds` |
| token | `budget.max_tokens` |
| 搜索空间 | `search.*` 是有限网格，耗尽后不再扩展 |
| 上线门控 | 整体 RMSE 须再降 ≥1%，且**单根轴承退化不超过 champion 全局 RMSE 的 20%** |

最后一条是关键：只优化全局均值会让最差的轴承崩掉，而隐藏测试流里
恰恰是短寿命轴承最危险。门控按**全局尺度**衡量局部退化。

### 5.3 大模型的角色

Qwen3.8-27B（或 PyTRiO 后训练模型）**只负责提出假设与动作建议**，
不直接改模型、不直接改门控。它的每次输出都过 `validate_plan()`：
动作必须在白名单内、`hypothesis` 非空、`risk` 在枚举内、数值在范围内。
任何解析失败、传输失败、schema 不符，都回落到确定性策略并在账本里标注。

### 5.4 退出码

| 码 | 含义 |
|---:|---|
| `0` | 最终 RMSE ≤ 官方基线 RMSE（通过基础门槛 ③） |
| `2` | 跑完了，但最终性能低于基线 |

评测机应当以 `0` 判定成功。**在断网、无端点的条件下，本实现返回 0。**

---

## 6. 资源占用

| 项 | 实测 |
|---|---|
| 依赖 | `numpy` 唯一硬依赖；`swanlab` 可选（不装也能跑） |
| 设备 | 纯 CPU，单进程 |
| 模型规模 | 3,457 参数（34→96→1） |
| 训练时长 | 一次闭环 9 个批次约 **75 秒**训练时间（`max_train_seconds` 默认 900，余量充足） |
| 峰值内存 | 数百 MB 级 |
| 磁盘 | 产物约 2 MB（含检查点） |

---

## 7. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `no such file: .../train.jsonl` | `RUL_DATA_ROOT` 没指向数据包根目录。容器里应是 `/data`，且 `public/train.jsonl` 存在 |
| `invalid checkpoint architecture` | 用了官方 loader 打开本系统的检查点。官方 loader 硬编码 `hidden ≤ 128` 与 `np.tanh`；本系统默认 `hidden=96` + tanh，可正常打开 |
| 退出码 `2` | 跑完了但性能低于基线。看 `run_report.json` 的 `final` 与 `baseline`，以及 `ledger.jsonl` 里哪一步被门控拒绝 |
| 决策全是 `do_nothing` | 漂移未触发或搜索空间耗尽。看 `ledger.jsonl` 的 `stage=monitor` 记录里的 `drift.level` |
| 日志里有 GBK 编码错误 | Windows 控制台。`harness/__init__.py` 已把 stdout/stderr 重配为 `errors="replace"`；若仍出现，说明绕过包导入直接跑了脚本 |

---

## 8. 基础门槛自检

赛题三条基础完成门槛，逐条对应：

| 门槛 | 自检方式 | 本实现实测 |
|---|---|---|
| ① 封闭评测中无人工干预完成全部阶段并按时输出结果 | `docker run --network none`，观察退出码与 `/out` 产物 | 断网 + 零参数运行，退出码 `0`，9 个批次全跑完 |
| ② 全部训练/调整/评估实验按规范完整留存 | `tools/verify_ledger.py` + `hypothesis_chain.jsonl` | 9 条账本记录哈希链完整；9 条证据链**四段全部完整** |
| ③ 封闭评测综合性能不低于官方静态基线 | `run_report.json` 的 `final.rmse` vs `baseline.rmse` | **283.54 vs 681.16（−58.4%）** |
