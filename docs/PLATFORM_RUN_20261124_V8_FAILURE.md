# 2026-11-24 平台试运行 v8：上游 504 失败与第二次运行

> 数据来源：平台「评测运行」页（`https://rsi-index.115.zone/#runs`），
> 通过工作台 API `/api/v1/runs`、`/api/v1/submissions`、`/api/v1/image-archives` 读取。
> 队伍 `team_id = addd5b126ab3d9e55a18df961593fd57`，账号 `gy_65511`。
>
> 与 2026-11-23 那次（提交版本 v6）逐轮对照。

---

## 0. 先说结论

**v8 的第一次运行失败在数据集 C 的第 6 轮，成因是上游模型网关返回 HTTP 504，
不是选手代码问题。** 平台自己在失败详情里写明：

> 上游模型服务暂时不可用或发生异常（上游 HTTP 504），该请求的处理结果或用量尚未确认。
> 这通常属于上游服务临时波动而非选手代码问题，平台已保留原账本核查，**不自动重发该请求**。

关键事实：

- **A、B 两个数据集 10 轮全部跑完**（`phase: completed`）—— 说明**新镜像本身能跑通**。
- 失败发生在 **C 的第 6/10 轮**，属于运行中途。
- **正式评测额度未消耗**：正式评测 **已用 0 / 3 次，剩余 3 次**；
  试运行已用 5 次（上限 8）。
- 失败后平台**没有**自动重发，用户手动发起了第二次运行（备注「第二次运行」）。

---

## 1. 镜像绑定：确认跑的是第五版（已含 C 方向根因修复）

平台侧镜像归档记录：

| 字段 | 值 |
|---|---|
| `image_archive_id` | `1c1c15536e00d43d129e7457fbb905b1` |
| 上传时间 | 2026-11-24 09:48:45 UTC = **17:48:45 GMT+8** |
| `size_bytes` | **182 732 800** |
| `status` | `ready` |

本地根目录 `zhixun-engine-linux-amd64.tar` 的字节数 **182 732 800**，与平台记录
**逐字节一致**。且该归档是在 2026-11-24 14:27 构建之后上传的。

> **结论：这次跑的确实是含 C 方向根因修复的镜像**（切片降级为记录、
> 判决只看全量 RMSE、相对退化度加权、epoch 预算 32–75）。

历史镜像归档对照（`/api/v1/image-archives`）：

| `id` | 上传时间 | 大小 | 对应提交 |
|---|---|---:|---|
| `1c1c1553…` | 11-24 09:48 | 182 732 800 | **v8（本次，含根因修复）** |
| `24d8e85c…` | 11-23 12:47 | 182 722 560 | v6（11-23 那次，56.25329） |
| `74c9ad8a…` | 11-22 12:06 | 195 706 880 | 未绑定提交 |
| `58cc7f50…` | 11-22 11:15 | 65 261 056 | v5（**当前正式选中**的提交绑的是这个） |
| `edb2d4db…` | 11-17 06:58 | 182 456 320 | v2 / v3 |

> ⚠️ **待办**：`selected: true` 的提交（v5）绑的仍是 `58cc7f50…`（62.2 MiB 的老包）。
> 正式评测前必须把**正式选中**的提交指向 `1c1c1553…`，否则正式跑的是没有
> C 方向修复的旧镜像。参见 `docs/平台提交表单填写指南.md`。

提交 `620b3c3a…`（v8）的字段：

| 字段 | 值 |
|---|---|
| `model_source` / `model` | `sampler` / `trio://run_7yqgoj5be2n0/sampler_weights/planner-27b-sft-v1` |
| `source_url` | `https://github.com/GY-phd/ai-` |
| `training_url` | `https://raw.githubusercontent.com/GY-phd/ai-/main/runs/harness-offline/run_report.json` |
| `image_archive_id` | `1c1c15536e00d43d129e7457fbb905b1` |
| `status` | `valid`（`selected: false`） |
| `document_id` | `c6e5e4e8b2fc4d360a58bd80d843e450` |

另有一个 `status: draft` 的提交 `542ab56d…`（10-12:18 创建，`image_archive_id` 同为
`1c1c1553…`），提交前需要确认平台最终采纳的是 `620b3c3a…`。

---

## 2. 失败详情的完整字段

```json
{
  "id": "ff819fccfff4268a3710294456064945",
  "mode": "trial",
  "submission_id": "620b3c3a28194f3856d183d11478659c",
  "submission_version": 8,
  "status": "failed",
  "eligible": false,
  "failure_kind": "unknown",
  "failure_detail_source": "worker",
  "created_at":   "2026-11-24T10:14:01.736706Z",
  "updated_at":   "2026-11-24T11:12:52.419454Z",
  "progress": {
    "parallel": true,
    "phase": "training",
    "round": 6, "round_total": 10,
    "datasets": {
      "A": {"phase": "completed"},
      "B": {"phase": "completed"},
      "C": {"phase": "training", "round": 6, "round_total": 10}
    }
  }
}
```

`failure_kind` 是 `unknown`（不是 `platform`），与 11-17 那次
`PREFLIGHT_INCOMPATIBLE`（`failure_kind: platform`）**性质不同**：
那次是镜像适配问题，这次是上游服务波动。

---

## 3. 为什么这次失败值得注意：我方代码在模型调用失败时会抛异常

`contest_baselines/layers/controller/round.py` 的 `run()` 顺序是固定的：

```python
self.open_parent()
self.read_public()
self.plan()
self.measure_parent()
self.build_candidate()
self.measure_candidate()
self.decide()
self.record()          # <- 测量与决策已经落账
self.consult_the_model()   # <- 模型复核；失败会抛
self.submit()
self.finish()
```

`_call_model()` 内部已经做了 3 次尝试（`MODEL_CALL_ATTEMPTS = 3`，
退避 `MODEL_CALL_BACKOFF = 3.0`），三次 payload 依次是：

1. `enable_thinking=False` + 1024 token
2. 不带该选项 + 1024 token
3. 不带该选项 + 平台给的完整 `max_output_tokens`

但文档字符串写得很明确：

> **The call is never skipped**: the round's contract is that it contains a real
> model call, so the last attempt is allowed to raise.

所以当上游网关持续 504 时，异常会从 `consult_the_model()` 抛出，
平台把该轮判为失败，**整轮运行随之失败**——即使此时 A、B 已经跑完、
C 的前 5 轮测量结果也已经写进账本。

**这不是 bug，是赛题的硬性契约**（赛题要求每轮至少一次真实模型调用）。
但它意味着：**上游一次 504 就能让整次试运行作废**，而平台**不会自动重发**。

因此可操作的对策是**外部的**：

- 失败后**手动重新发起**试运行（本次已于 19:16:10 发起第二次运行）；
- 正式评测只有 3 次额度，每次都要挑上游服务健康的时候提交。

---

## 4. 第二次运行

| 字段 | 值 |
|---|---|
| `id` | `447ee92c039928cf1c170950e6c7e61b` |
| `created_at` | 2026-11-24T11:16:10Z = **19:16:10 GMT+8** |
| `trial_note` | 「第二次运行」 |
| `submission_id` / `submission_version` | `620b3c3a…` / **8**（同一个提交、同一个镜像） |
| `status` | **`failed`**（同样 504） |
| `updated_at` | 2026-11-24T11:25:02Z = **19:25:02 GMT+8** |

失败到重试间隔 **3 分 18 秒**；第二次运行**只跑了约 8 分 52 秒**就在
**第 1 轮**失败（A、B、C 都停在 `round 1/10`），比第一次
（58 分 51 秒、A/B 各跑完 10 轮、C 死在第 6 轮）失败得更快。

**连续两次同样原因失败 ⇒ 上游当前处于持续劣化状态，不是偶发波动。**

---

## 4.1 本地直接探测 PyTRiO：504 的真正来源

从本地用同一把 API Key 直接调用 PyTRiO（`pytrio.ServiceClient`）：

| 探测项 | 结果 |
|---|---|
| `get_server_capabilities()` | ✅ 正常，返回受支持模型列表 |
| 受支持的**基础**模型 | `Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-4B:peft:256k`、`Qwen/Qwen3.8-27B` |
| `Qwen/Qwen3.8-27B` 采样 | ⚠️ **成功但极慢**：64 token 的请求分别耗时 **141.6 s** 与 **825.4 s**（两次实测），生成 3 个 output token |
| **`trio://run_7yqgoj5be2n0/sampler_weights/planner-27b-sft-v1`** | ❌ **`model_not_found`（HTTP 404）** |

两个独立问题：

### (1) Sampler 路径当前已不可见

```
ControlAPIError: model not found
Hint: The requested model is not published or not visible to this workspace.
Details: {'code': 'model_not_found', 'status': 404}
```

提交表单里填的 `model` 字段正是这条 `trio://` 路径（`model_source: sampler`）。
它**不在** `get_supported_models()` 的列表里 —— 受支持列表里只有三个基础模型，
没有任何 `trio://` 条目。

> ⚠️ **这与 11-22 那次 v5 试运行的实测结论矛盾**：当时实测该 `trio://` 路径
> 「真实存在且可调用」（见 `docs/PLATFORM_REPORT_REVIEW.md` §5）。
> 现在从本地直接探测返回 404。可能是权重已被清理、租约到期，
> 或该路径只在平台工作区可见、而不在本地账号可见。
> **必须以平台网关的视角复核**，因为正式评测时是平台去调这条路径。

### (2) 基础模型 27B 慢到会触发网关 504

一次 64 token 的请求要 **141.6 ~ 825.4 秒**。平台网关的 504 就是在这个
等待窗口里超时返回的 —— 我们的 `_call_model()` 只有 3 次尝试
（退避 3 s），每次尝试都要等这么久，3 次都拿不到结果就抛异常，
整轮运行随之失败。

**推论**：即使平台把请求转给了基础模型 27B，只要单次响应超过网关的超时窗口，
就会稳定地产生 504。这解释了为什么两次运行都在**模型调用阶段**失败，
且第二次只跑了 8 分 52 秒（第一轮的第一模型调用就没扛住）。

---

## 5. 给正式评测前的检查清单

1. **把正式选中的提交指向新镜像** `1c1c15536e00d43d129e7457fbb905b1`，
   而不是仍在 `selected: true` 上的 v5（`58cc7f50…`，62.2 MiB 老包）。
2. **确认 `selected` 的那一条的 `training_url` 指向
   `runs/harness-offline/run_report.json`**（v5 现在指向的是
   `docs/PLANNER_SFT_RECORD.md`）。
3. **正式评测的 3 次额度要留到上游稳定时用**。上游 504 不计正式额度
   （本次失败发生在试运行阶段），但正式评测期间失败会浪费一次。
4. 每次试运行失败后**手动重发**，不要等平台自动重试（平台明确不重发）。
5. **复核 `trio://` sampler 路径是否仍可调用**。本地探测已返回
   `model_not_found`（§4.1 (1)）。若它确实不可用了，按
   `docs/平台提交表单填写指南.md` 的预案**改选「基础模型」**
   （`Qwen/Qwen3.8-27B`），并在表单里如实说明未使用微调权重。
   基础模型同样可以正式参赛。
6. **上游 27B 单次响应要 141~825 秒**，远超一般网关超时窗口。
   若平台网关的超时不可调，则每一次模型调用都是一发"赌"。
   这是**平台侧**的问题（`failure_detail_source: worker`、
   `failure_kind: unknown`），但会持续打掉我们的试运行次数，
   需要**联系管理员**核查账本与网关超时设置。

---

## 6. 当前状态小结（2026-11-24 19:30）

| 项 | 状态 |
|---|---|
| 正式评测额度 | **0 / 3 已用，剩余 3** |
| 试运行额度 | 5 / 8 已用（v8 两次都失败） |
| 镜像 | ✅ `1c1c1553…` 与本地第五版字节数一致 |
| 代码 | ✅ 已推送 `origin/main`（`70308a3`） |
| 上游 | ❌ 连续两次 504；本地探测基础模型 27B 慢至 141~825 s |
| Sampler 路径 | ⚠️ 本地探测 404 `model_not_found`，需平台视角复核 |
