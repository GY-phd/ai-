"""漂移监测与归因。

- monitor : 无标签在线漂移检测 (PSI + KS), 判断输入分布是否漂移
- attribution: 结合标签与误差, 区分协变量漂移 / 概念漂移, 并定位到特征
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .data import Dataset


# =====================================================================
# PSI (Population Stability Index)
# =====================================================================

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    min_pct: float = 1e-6,
) -> float:
    """PSI = Σ (实际占比 - 期望占比) * ln(实际占比 / 期望占比)。

    reference/current 均为 1D 数组 (连续特征)。分箱边界由 reference 分位数决定。
    """
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 2 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), min_pct, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), min_pct, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_psi_table(
    reference: Dataset,
    current: Dataset,
    bins: int = 10,
) -> pd.DataFrame:
    """逐特征 PSI 表。"""
    rows = []
    for i, name in enumerate(reference.feature_names):
        psi = compute_psi(reference.X[:, i], current.X[:, i], bins=bins)
        rows.append({"feature": name, "psi": psi})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def ks_test_pvalue(ref: np.ndarray, cur: np.ndarray) -> float:
    """双样本 KS 检验 p 值。"""
    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return 1.0
    try:
        _, p = stats.ks_2samp(ref, cur)
    except Exception:
        return 1.0
    return float(p)


# =====================================================================
# 漂移事件
# =====================================================================

@dataclass
class DriftReport:
    """一次漂移检测的完整结论。"""

    triggered: bool
    severity: str  # none | warn | alarm
    overall_psi: float
    max_psi: float
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    drifted_features: List[str] = field(default_factory=list)
    ks_hits: List[str] = field(default_factory=list)
    concept_drift: bool = False
    label_coverage: float = 0.0
    error_rate: Optional[float] = None
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "triggered": self.triggered,
            "severity": self.severity,
            "overall_psi": round(self.overall_psi, 5),
            "max_psi": round(self.max_psi, 5),
            "top_features": [[n, round(v, 5)] for n, v in self.top_features],
            "drifted_features": self.drifted_features,
            "ks_hits": self.ks_hits,
            "concept_drift": self.concept_drift,
            "label_coverage": round(self.label_coverage, 4),
            "error_rate": None if self.error_rate is None else round(self.error_rate, 5),
            "details": {
                k: (round(v, 6) if isinstance(v, (int, float)) else v)
                for k, v in self.details.items()
            },
        }


class DriftMonitor:
    """在线漂移监测器。

    用参考集 (训练集/最近一次已知良好分布) 作为 baseline, 对流窗口做检测。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.psi_warn = float(cfg.get("drift.psi_warn", 0.1))
        self.psi_alarm = float(cfg.get("drift.psi_alarm", 0.25))
        self.ks_pvalue = float(cfg.get("drift.ks_pvalue", 0.01))
        self.bins = int(cfg.get("drift.psi_bins", 10))
        self.min_samples = int(cfg.get("drift.min_samples_for_action", 300))
        self.min_label_coverage = float(cfg.get("attribution.min_label_coverage", 0.2))

    def check(
        self,
        reference: Dataset,
        current: Dataset,
        predictions: Optional[np.ndarray] = None,
    ) -> DriftReport:
        n = len(current)
        if n < self.min_samples:
            return DriftReport(
                triggered=False,
                severity="none",
                overall_psi=0.0,
                max_psi=0.0,
                details={"reason": "insufficient_samples", "n": float(n)},
            )

        psi_table = feature_psi_table(reference, current, bins=self.bins)
        overall = float(psi_table["psi"].mean())
        max_psi = float(psi_table["psi"].max())
        drifted = psi_table.loc[psi_table["psi"] >= self.psi_warn, "feature"].tolist()
        top = [(str(r.feature), float(r.psi)) for r in psi_table.head(5).itertuples()]

        ks_hits: List[str] = []
        for i, name in enumerate(reference.feature_names):
            if psi_table.loc[psi_table["feature"] == name, "psi"].iloc[0] >= self.psi_warn:
                if ks_test_pvalue(reference.X[:, i], current.X[:, i]) < self.ks_pvalue:
                    ks_hits.append(name)

        severity = "none"
        if max_psi >= self.psi_alarm or overall >= self.psi_alarm:
            severity = "alarm"
        elif drifted:
            severity = "warn"

        # ---- 概念漂移判定: 有标签时看误差是否显著上升 ----
        concept = False
        label_cov = 0.0
        err = None
        task = str(self.cfg.get("data.task", "binary"))
        if current.has_labels and predictions is not None:
            label_cov = 1.0
            y = current.y
            pred = predictions
            if task == "regression":
                # 回归任务的误差是绝对误差 (分钟), 不能用 `pred != y` 判断 ——
                # 那会把"预测值略有偏差"也算作完全错误, 概念漂移判定会彻底失真。
                err = float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))))
            else:
                err = float(np.mean(pred != y))
            # 与参考集上的同口径误差比较
            base_pred = getattr(self, "_baseline_error", None)
            if base_pred is not None and base_pred > 0:
                concept = err > base_pred * 1.25
        elif current.has_labels:
            label_cov = 1.0

        triggered = severity in ("warn", "alarm") and (
            concept or max_psi >= self.psi_alarm or len(drifted) >= max(1, len(drifted))
        )
        # 无标签场景下只看分布类漂移
        if not current.has_labels:
            triggered = severity in ("warn", "alarm")

        return DriftReport(
            triggered=triggered,
            severity=severity,
            overall_psi=overall,
            max_psi=max_psi,
            top_features=top,
            drifted_features=drifted,
            ks_hits=ks_hits,
            concept_drift=concept,
            label_coverage=label_cov,
            error_rate=err,
            details={"n": float(n), "n_drifted": float(len(drifted))},
        )

    def set_baseline_error(self, err: float) -> None:
        self._baseline_error = float(err)


# =====================================================================
# 归因分析
# =====================================================================

@dataclass
class Attribution:
    """漂移归因结论。"""

    drift_type: str  # covariate | concept | mixed | unknown
    culprit_features: List[str]
    rationale: str
    evidence: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "drift_type": self.drift_type,
            "culprit_features": self.culprit_features,
            "rationale": self.rationale,
            "evidence": {k: round(float(v), 6) for k, v in self.evidence.items()},
        }


class DriftAttributor:
    """把漂移事件归因为具体类型与具体特征。

    判定逻辑:
      * PSI 显著 + 误差未显著上升        -> covariate (输入分布漂移)
      * PSI 不显著 + 误差显著上升        -> concept   (输入-输出关系漂移)
      * 两者都显著                        -> mixed
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_bins = int(cfg.get("attribution.n_error_bins", 5))
        self.psi_alarm = float(cfg.get("drift.psi_alarm", 0.25))
        self.psi_warn = float(cfg.get("drift.psi_warn", 0.1))
        self.margin = float(cfg.get("safety.rollback_margin", 0.005))
        # 误差上升的绝对门限: 处理基线误差接近 0 (过拟合/易任务) 时的退化情形
        self.abs_error_margin = float(cfg.get("attribution.abs_error_margin", 0.03))

    def attribute(
        self,
        report: DriftReport,
        baseline_error: Optional[float],
        recent_error: Optional[float],
    ) -> Attribution:
        """判定漂移类型。

        判据 (双轨: 相对 + 绝对, 避免小基线误差导致漏判):
          * 协变量: max PSI ≥ alarm (强) 或 ≥ warn (弱)
          * 概念:   误差相对上升 > 20% 且绝对上升 > 门限
        """
        cov_strong = report.max_psi >= self.psi_alarm
        cov_weak = report.max_psi >= self.psi_warn

        err_increase = 0.0
        con = False
        if baseline_error is not None and recent_error is not None:
            err_increase = recent_error - baseline_error
            rel_growth = err_increase > max(0.0, baseline_error) * 0.2
            abs_growth = err_increase > self.abs_error_margin
            con = bool(abs_growth and (rel_growth or baseline_error < 0.01))
            # 基线仍可参考时, 即便误差小也要求同向增长
            if baseline_error >= 0.01 and not rel_growth:
                con = False
        elif report.concept_drift:
            con = True

        if cov_strong and con:
            dtype = "mixed"
        elif cov_strong:
            dtype = "covariate"
        elif con:
            dtype = "concept"
        elif cov_weak:
            dtype = "covariate_weak"
        else:
            dtype = "unknown"

        culprits = report.drifted_features[:5]
        rationale = self._rationale(dtype, report, baseline_error, recent_error)
        return Attribution(
            drift_type=dtype,
            culprit_features=culprits,
            rationale=rationale,
            evidence={
                "max_psi": report.max_psi,
                "overall_psi": report.overall_psi,
                "baseline_error": baseline_error or 0.0,
                "recent_error": recent_error or 0.0,
                "error_increase": err_increase,
            },
        )

    @staticmethod
    def _rationale(
        dtype: str,
        report: DriftReport,
        base_err: Optional[float],
        cur_err: Optional[float],
    ) -> str:
        base = base_err or 0.0
        cur = cur_err or 0.0
        if dtype == "covariate":
            return (
                f"输入分布发生显著偏移 (max PSI={report.max_psi:.3f}), "
                f"主要特征 {', '.join(report.drifted_features[:3]) or '未定位'}, "
                f"误差 {base:.3f}→{cur:.3f} 变化有限, 判定为协变量漂移, 建议重训适配新分布。"
            )
        if dtype == "concept":
            return (
                f"误差从 {base:.3f} 升至 {cur:.3f}, 而分布偏移有限 "
                f"(max PSI={report.max_psi:.3f}), 判定为概念漂移 (输入-输出关系改变), "
                f"建议以最新标签重训。"
            )
        if dtype == "mixed":
            return (
                f"同时出现分布偏移 (max PSI={report.max_psi:.3f}) 与误差上升 "
                f"({base:.3f}→{cur:.3f}), 判定为混合漂移, 建议重训 + 增量样本加权。"
            )
        if dtype == "covariate_weak":
            return (
                f"输入分布出现轻度偏移 (max PSI={report.max_psi:.3f}), "
                f"误差 {base:.3f}→{cur:.3f} 尚未显著恶化, 建议持续观察, 暂不更新。"
            )
        return "证据不足, 建议继续观察一个窗口后再决策。"


def error_by_feature_bin(
    ds: Dataset,
    predictions: np.ndarray,
    feature_idx: int,
    n_bins: int = 5,
) -> Dict[str, float]:
    """按某特征分箱统计误差率, 用于判断"局部退化"型漂移。"""
    x = ds.X[:, feature_idx]
    if len(x) == 0:
        return {}
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return {}
    edges = edges.astype(np.float64)
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(x, edges[1:-1])
    out: Dict[str, float] = {}
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        out[f"bin{b}"] = float(np.mean(predictions[m] != ds.y[m]))
    return out
