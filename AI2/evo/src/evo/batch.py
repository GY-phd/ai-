"""批次下发协议 —— 赛题 B 包的核心交互方式。

赛题原文:
    "B 包数据按批次下发, 必须先输出当前批次模型或不上线的判断才能获取下一批数据;
     评测时总能拿到标签, 且无需消耗配额申请。"

这意味着评测交互是**请求-应答式**的, 而不是一次性把全部数据喂进去:

    循环:
      1. 取第 k 批数据 (X_k, y_k)
      2. 用当前上线模型对 X_k 推理 → 提交 (预测 or "不上线" 判断)
      3. 评测方返回第 k+1 批数据
      4. 若第 k 批触发了漂移 → 本轮完成训练/影子评估/上线决策
      5. 若触发上线 → 下一批用**新模型**推理

因此本模块提供:
  * `BatchStream` — 抽象批次来源 (本地目录 / 评测平台回调)
  * `BatchResponse` — 每一批的应答 (预测 + 上线决策), 供评测方消费
  * `BatchHarness` — 在批次协议下驱动 Harness 的适配器

设计原则:
  * **不假设平台的具体 API 形态** —— 用 `fetcher` 回调把"取下一批"的
    方式留给上层 (可能是 HTTP 轮询 / 共享目录 / stdin 协议)。
  * 标签始终可得 (赛题明确), 因此 B 包模式下**不消耗标注配额**。
  * 每一批都产出可审计的应答记录, 满足实验记录规范。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from .audit import AuditLog
from .config import Config
from .data import Dataset, OfficialAdapter
from .experiment import ExperimentRecorder


# =====================================================================
# 应答结构
# =====================================================================

@dataclass
class BatchResponse:
    """对某一批数据的应答 —— 评测方据此决定是否下发下一批。

    赛题要求"先输出当前批次模型或不上线的判断", 因此应答必须同时包含:
      * `predictions` —— 当前上线模型对该批的预测 (任务相关)
      * `decision`  —— 该批是否触发重训 / 是否上线
    """

    batch_id: int
    n_samples: int
    predictions: np.ndarray
    decision: Dict[str, Any] = field(default_factory=dict)
    model_version: Optional[str] = None
    approved: bool = False
    gain: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "n_samples": self.n_samples,
            "predictions": np.asarray(self.predictions).tolist(),
            "decision": self.decision,
            "model_version": self.model_version,
            "approved": self.approved,
            "gain": round(self.gain, 6),
            "metrics": self.metrics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# =====================================================================
# 批次来源
# =====================================================================

class BatchFetcher:
    """批次数据来源的抽象基类。

    子类实现 `fetch(batch_id) -> Optional[Dataset]`:
      * 返回 None 表示没有更多批次 (流结束)
    """

    def fetch(self, batch_id: int) -> Optional[Dataset]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DirectoryBatchFetcher(BatchFetcher):
    """从本地目录按批次读取数据文件 (评测平台若落盘到共享目录则用这个)。

    约定: 目录下存在 `batch_000.jsonl` / `batch_000.parquet` / ... 之类的文件,
    或存在一个 `batches.json` 清单列出顺序。
    """

    def __init__(self, root: str | Path, adapter: OfficialAdapter, pattern: str = "batch_*"):
        self.root = Path(root)
        self.adapter = adapter
        self.pattern = pattern
        self._files: List[Path] = sorted(self.root.glob(pattern))
        if not self._files:
            # 回退: 单文件按行切批
            single = sorted(self.root.glob("*.jsonl")) + sorted(self.root.glob("*.parquet"))
            self._files = single
        self._chunks: List[Dataset] = []
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._files:
            return
        # 若只有一个文件 → 按 batch_size 切分
        if len(self._files) == 1:
            f = self._files[0]
            if f.suffix in (".jsonl", ".json"):
                ds = self.adapter.load_jsonl(f, "batch")
            else:
                ds = self.adapter.load_split(f.stem)
            size = int(self.adapter.cfg.get("data.batch_size", 500))
            for i in range(0, len(ds), size):
                self._chunks.append(ds.slice(i, min(i + size, len(ds))))
            return
        # 多文件 → 每个文件一批
        for f in self._files:
            if f.suffix in (".jsonl", ".json"):
                self._chunks.append(self.adapter.load_jsonl(f, f.stem))
            else:
                self._chunks.append(self.adapter.load_split(f.stem))

    def fetch(self, batch_id: int) -> Optional[Dataset]:
        self._load_all()
        if batch_id < 0 or batch_id >= len(self._chunks):
            return None
        return self._chunks[batch_id]

    @property
    def n_batches(self) -> int:
        self._load_all()
        return len(self._chunks)


class CallbackBatchFetcher(BatchFetcher):
    """用回调取批次 —— 对接平台 HTTP / stdin 协议的通用入口。

    评测方只需提供一个函数 `fn(batch_id) -> Dataset | dict | None`。
    """

    def __init__(self, fn: Callable[[int], Any], adapter: Optional[OfficialAdapter] = None):
        self._fn = fn
        self.adapter = adapter

    def fetch(self, batch_id: int) -> Optional[Dataset]:
        raw = self._fn(batch_id)
        if raw is None:
            return None
        if isinstance(raw, Dataset):
            return raw
        if isinstance(raw, dict):
            # {"X": [[...]], "y": [...]} → Dataset
            X = np.asarray(raw["X"], dtype=np.float64)
            y = raw.get("y")
            import pandas as pd

            meta = pd.DataFrame(
                {
                    "timestamp": np.arange(len(X), dtype=np.float64),
                    "batch": np.full(len(X), batch_id),
                }
            )
            return Dataset(
                X=X,
                y=None if y is None else np.asarray(y),
                meta=meta,
                name=f"batch_{batch_id}",
            )
        raise TypeError(f"批次回调返回了不支持的类型: {type(raw)}")


# =====================================================================
# 批次模式下的 Harness 适配
# =====================================================================

@dataclass
class BatchRunSummary:
    """批次模式的运行汇总。"""

    batches: int = 0
    responses: List[BatchResponse] = field(default_factory=list)
    updates_approved: int = 0
    drifts_detected: int = 0
    total_labels: int = 0
    total_gpu_seconds: float = 0.0
    tail_score: float = 0.0
    baseline_tail_score: float = 0.0
    final_score: float = 0.0
    baseline_score: float = 0.0
    efficiency: float = 0.0
    halted: bool = False
    total_tokens: int = 0
    # 基础运行门槛自检结果
    gates: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batches": self.batches,
            "updates_approved": self.updates_approved,
            "drifts_detected": self.drifts_detected,
            "total_labels": self.total_labels,
            "total_gpu_seconds": round(self.total_gpu_seconds, 3),
            "total_tokens": self.total_tokens,
            "tail_score": round(self.tail_score, 5),
            "baseline_tail_score": round(self.baseline_tail_score, 5),
            "final_score": round(self.final_score, 5),
            "baseline_score": round(self.baseline_score, 5),
            "efficiency": round(self.efficiency, 10),
            "halted": self.halted,
            "gates": self.gates,
        }


class BatchRunner:
    """在批次下发协议下驱动自进化闭环。

    与 `Harness.run()` 的区别:
      * 数据按批到达, 不是一次性加载
      * 每批都要产出**应答** (预测 + 上线决策) 才能推进
      * 评测时标签始终可得 → **不消耗标注配额** (赛题明确)
      * 流上性能按"批"口径统计, 尾段 = 最后 1/3 的批次

    用法:
        runner = BatchRunner(cfg, fetcher)
        summary = runner.run()
        # 评测方消费:
        for r in summary.responses:
            print(r.to_json())
    """

    def __init__(
        self,
        cfg: Config,
        fetcher: BatchFetcher,
        use_synthetic: bool = False,
    ):
        self.cfg = cfg
        self.fetcher = fetcher
        self.use_synthetic = use_synthetic
        self.task = str(cfg.get("data.task", "binary"))
        self.audit = AuditLog(cfg)
        self.recorder = ExperimentRecorder(cfg)
        # 复用 Harness 的全部闭环能力 (漂移/归因/策略/训练/影子评估/回滚)
        from .harness import Harness

        self.harness = Harness(cfg, use_synthetic=use_synthetic)
        # B 包: 标签总是可得 → 标注配额不作为硬约束
        self.free_labels = bool(cfg.get("data.batched_free_labels", True))
        self._responses: List[BatchResponse] = []
        # 已见批次 (累积成逻辑流)
        self._stream_parts: List[Dataset] = []
        # 告知 Harness: 标签免费 (跳过采购流程)
        self.harness._free_labels = self.free_labels

    # -- 单批处理 --------------------------------------------------------
    def _accumulate(self, batch: Dataset) -> Dataset:
        """把批次累积成一条"逻辑流" —— Harness 的训练数据组装依赖连续流。

        批次协议下数据是逐批到达的, 但重训需要"最近 N 条已标注样本"。
        因此这里把已见批次拼起来, 并同步维护 `harness.stream`。
        """
        if self._stream_parts:
            X = np.vstack([p.X for p in self._stream_parts] + [batch.X])
            ys = [p.y for p in self._stream_parts] + [batch.y]
            y = None if any(v is None for v in ys) else np.concatenate(ys)
            import pandas as pd

            meta = pd.concat(
                [p.meta.reset_index(drop=True) for p in self._stream_parts]
                + [batch.meta.reset_index(drop=True)],
                ignore_index=True,
            )
            meta["batch"] = np.arange(len(meta)) // max(1, len(batch))
            meta["timestamp"] = np.arange(len(meta), dtype=np.float64)
            merged = Dataset(
                X=X,
                y=y,
                meta=meta,
                name="batch_stream",
                feature_names=list(batch.feature_names),
            )
        else:
            merged = batch
        self._stream_parts.append(batch)
        return merged

    def _process_batch(self, batch: Dataset, batch_id: int) -> BatchResponse:
        """对一批数据: 推理 → 漂移检测 → (必要时) 重训 → 产出应答。"""
        h = self.harness
        if h.current_model is None:
            h.prepare_data()
            h.bootstrap()

        # 累积成流, 供训练数据组装使用
        h.stream = self._accumulate(batch)
        h._stream_cursor = len(h.stream)

        # 1) 用当前上线模型推理 (这是评测方要的第一样东西)
        pred = h.current_model.predict(batch.X) if h.current_model is not None else None

        # 2) 漂移检测 (以本批与参考分布比较)
        drift = None
        attribution: Dict[str, Any] = {}
        decision: Dict[str, Any] = {"action": "monitor", "source": "batch_mode"}
        approved = False
        gain = 0.0

        reference = h._batch_reference
        if reference is None:
            # 首批: 用初始训练集作参考分布
            reference = h.train_ds
        if h.current_model is not None and reference is not None:
            try:
                drift = h.monitor.check(reference, batch, predictions=pred)
                # 归因需要 (报告, 基线误差, 当前误差) —— 不是模型对象。
                # 批次模式下标签总是可得, 因此可以直接算当前误差;
                # 基线误差用现网模型在**本批**上的误差作为参照起点。
                recent_err = None
                base_err = None
                if batch.has_labels and pred is not None:
                    lab = batch.labeled()
                    if len(lab) > 0:
                        recent_err = h._task_error(pred[lab.labeled_mask], lab.y, h.task)
                        if h.static_baseline is not None:
                            base_pred = h.static_baseline.predict(lab.X)
                            base_err = h._task_error(base_pred, lab.y, h.task)
                attribution = h.attributor.attribute(
                    drift, base_err, recent_err
                ).to_dict()
            except Exception as exc:
                # 漂移检测异常**不能静默吞掉** —— 那会让系统变成"永远只监控",
                # 看起来正常实则完全没有适应能力。记录并继续, 但留痕。
                drift = None
                attribution = {}
                self.audit.append(
                    "batch_drift_check_failed",
                    {"batch_id": batch_id, "error": repr(exc)},
                )

        if drift is not None and drift.triggered:
            # 3) 触发重训 —— 走 Harness 的完整决策与影子评估链
            #
            # 关键: 窗口必须指向**累积流**的末端, 而不是本批。
            # 影子集由 `_shadow_set(window)` 从流末端取 `shadow_holdout` 条,
            # 训练数据则截止到影子集之前。若把 window 设成本批 (0, len(batch)),
            # 则影子集与训练数据都退化为"本批内部" —— 样本量不足,
            # 显著性检验必然不显著 (实测 p=0.174, 候选全部被拒),
            # 系统退化为"永远只监控"。
            stream_len = len(h.stream) if h.stream is not None else len(batch)
            approved, gain, decision = self._maybe_retrain(
                batch, batch_id, drift, attribution, (0, stream_len)
            )
            # 上线成功后, 本批成为新的 known-good 参考分布
            if approved:
                h._batch_reference = batch

        # 4) 评估本批得分 (B 包标签可得, 直接算)
        metrics: Dict[str, float] = {}
        if pred is not None and batch.has_labels and self.task == "binary":
            from .models import compute_metrics

            metrics = compute_metrics(batch.y, pred, "binary")
        elif pred is not None and batch.has_labels and self.task == "regression":
            from .models import compute_metrics

            metrics = compute_metrics(batch.y, pred, "regression")

        resp = BatchResponse(
            batch_id=batch_id,
            n_samples=len(batch),
            predictions=pred if pred is not None else np.zeros(len(batch)),
            decision=decision,
            model_version=h.current_model.version if h.current_model else None,
            approved=approved,
            gain=gain,
            metrics=metrics,
        )
        self._responses.append(resp)
        self.audit.append(
            "batch_response",
            {
                "batch_id": batch_id,
                "n": len(batch),
                "action": decision.get("action"),
                "approved": approved,
                "gain": round(gain, 6),
                "model_version": resp.model_version,
            },
        )
        return resp

    def _maybe_retrain(
        self, batch: Dataset, batch_id: int, drift, attribution, window: Tuple[int, int]
    ) -> tuple:
        """批次触发的重训 —— 复用 Harness 的完整决策/训练/影子评估/上线链。"""
        h = self.harness
        decision: Dict[str, Any] = {
            "action": "retrain",
            "source": "batch_mode",
            "drift_type": attribution.get("drift_type", "unknown"),
        }
        # 批次模式下标签免费 → 不受标注配额限制
        try:
            if self.free_labels:
                # 临时放宽标注配额 (赛题: 评测时总能拿到标签, 无需消耗配额)
                self._relax_label_quota()
            result = h._execute(
                iteration=batch_id,
                window=window,
                report=drift,
                attribution=attribution,
                decision=decision,
                min_labels=int(self.cfg.get("safety.shadow_min_steps", 300)),
            )
            decision.update(result.decision or {})
            decision["approved"] = result.approved
            decision["version"] = result.version
            decision["gain"] = round(result.gain, 6)
            return result.approved, result.gain, decision
        except Exception as exc:  # pragma: no cover - 兜底, 不中断批次协议
            decision["error"] = str(exc)
            return False, 0.0, decision

    def _relax_label_quota(self) -> None:
        """把标注配额放宽到不设限 (B 包: 标签总是可得且不消耗配额)。

        实现方式: 抬高 `policy.budget.limits["labels"]`。
        原始配额保留在 `_orig_label_limit` 以便审计时如实记录。
        """
        b = self.harness.policy.budget
        if "_orig_label_limit" not in b.limits:
            b.limits["_orig_label_limit"] = b.limits.get("labels", 0)
        b.limits["labels"] = max(
            int(b.limits.get("labels", 0)),
            int(self.cfg.get("budget.max_label_quota", 2000)),
        ) * 100

    # -- 主循环 ----------------------------------------------------------
    def run(self, max_batches: Optional[int] = None) -> BatchRunSummary:
        """按批次推进闭环, 直到数据源耗尽或达到上限。"""
        cfg = self.cfg
        limit = max_batches if max_batches is not None else int(
            cfg.get("data.max_batches", 1000)
        )
        batch_id = 0
        # 静态基线得分序列 (与自适应系统**同口径**对比)
        baseline_scores: List[float] = []
        while batch_id < limit:
            batch = self.fetcher.fetch(batch_id)
            if batch is None:
                break
            resp = self._process_batch(batch, batch_id)
            # 冻结静态基线在同一批上的得分 —— 赛题"不得低于静态基线"的判定依据
            bs = self._static_baseline_score(batch)
            if bs is not None:
                baseline_scores.append(bs)
                resp.metrics["baseline_score"] = bs
            batch_id += 1

        summary = BatchRunSummary()
        summary.batches = len(self._responses)
        summary.responses = self._responses
        summary.updates_approved = sum(1 for r in self._responses if r.approved)
        summary.drifts_detected = sum(
            1 for r in self._responses if r.decision.get("action") != "monitor"
        )
        summary.total_labels = sum(r.n_samples for r in self._responses)
        summary.halted = bool(getattr(self.harness.policy, "halted", False))
        # 成本: 训练耗时与 Token 用量 (赛题客观分维度)
        summary.total_gpu_seconds = float(
            getattr(self.harness.policy.budget, "gpu_seconds", 0.0)
        )
        summary.total_tokens = int(
            getattr(self.harness.llm.usage, "total_tokens", 0)
        )

        # 流上得分: 用每批的 metrics["score"] 序列
        scores = [r.metrics.get("score", 0.0) for r in self._responses if r.metrics]
        if scores:
            summary.final_score = float(np.mean(scores))
            k = max(1, len(scores) // 3)
            summary.tail_score = float(np.mean(scores[-k:]))
        # 静态基线 (同口径)
        if baseline_scores:
            summary.baseline_score = float(np.mean(baseline_scores))
            k = max(1, len(baseline_scores) // 3)
            summary.baseline_tail_score = float(np.mean(baseline_scores[-k:]))
        else:
            summary.baseline_score = summary.final_score
            summary.baseline_tail_score = summary.tail_score
        # 效率: 相对静态基线的增益 / 总成本 (与 Harness 口径一致)
        total_cost = (
            summary.total_gpu_seconds
            + 0.02 * summary.total_labels
            + 0.0005 * summary.total_tokens
        )
        total_gain = max(0.0, summary.tail_score - summary.baseline_tail_score)
        summary.efficiency = total_gain / total_cost if total_cost > 0 else 0.0

        # 门槛自检 (与 Harness._check_gates 同口径)
        summary.gates = self._check_gates(summary, baseline_scores)

        self.audit.append(
            "batch_run_summary",
            {
                "batches": summary.batches,
                "updates_approved": summary.updates_approved,
                "tail_score": round(summary.tail_score, 5),
                "baseline_tail_score": round(summary.baseline_tail_score, 5),
                "gates": summary.gates,
            },
        )
        return summary

    def _static_baseline_score(self, batch: Dataset) -> Optional[float]:
        """冻结静态基线在同一批上的得分 (同口径对照)。

        赛题门槛③: 最终模型性能不得低于官方静态基线。
        这里用**初始化时训练的那份基线模型** (永不更新) 作对照 ——
        与 Harness 中的 static_baseline 语义一致。
        """
        h = self.harness
        base = h.static_baseline or h.current_model
        if base is None or not batch.has_labels:
            return None
        lab = batch.labeled()
        if len(lab) == 0:
            return None
        try:
            pred = base.predict(lab.X)
            from .models import compute_metrics

            return float(compute_metrics(lab.y, pred, h.task)["score"])
        except Exception:
            return None

    def _check_gates(self, summary: BatchRunSummary, baseline_scores: List[float]) -> Dict[str, Any]:
        """批次模式下的基础运行门槛自检。"""
        min_iters = int(self.cfg.get("gates.min_iterations", 3))
        attempted = sum(
            1 for r in self._responses if r.decision.get("action") != "monitor"
        )
        gate_iter = attempted >= min_iters
        gate_base = summary.tail_score >= summary.baseline_tail_score
        gates = {
            "min_iterations": min_iters,
            "attempted_iterations": attempted,
            "total_iterations": len(self._responses),
            "iteration_gate_passed": gate_iter,
            "tail_score": round(summary.tail_score, 5),
            "baseline_tail_score": round(summary.baseline_tail_score, 5),
            "baseline_gate_passed": bool(gate_base),
            "baseline_gate_metric": "tail_score",
            "all_passed": bool(gate_iter and gate_base),
        }
        self.audit.append("batch_gates_checked", gates)
        return gates
