"""Drift detection from public batch statistics.

Three independent signals are combined:

* performance drift -- the champion's batch RMSE versus its own reference RMSE;
* static-baseline drift -- the champion's batch RMSE versus the untouched
  official checkpoint on the same batch;
* data drift -- feature means/stds versus the checkpoint's frozen normalisation
  statistics, plus target-mean shift versus the frozen target scaler.

All calculations use only the batch's public validation rows and the
checkpoint's stored statistics.  No hidden data is read and no labels are fed
back into the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np

from .config import DriftConfig

SEVERITY_RANK = {"stable": 0, "warning": 1, "critical": 2}


def _severity(rank: int) -> str:
    return ("stable", "warning", "critical")[rank]


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-12))


def describe_feature(path: str) -> str:
    """Turn the baseline's JSON path encoding into ``sensors[3]`` style text."""
    try:
        parts = json.loads(path)
        text = ""
        for index, (kind, value) in enumerate(parts):
            if kind == "key":
                text += ("." if text and index else "") + str(value)
            elif kind == "index":
                text += f"[{value}]"
            else:
                return path
        return text or path
    except Exception:  # noqa: BLE001 - presentation only
        return path


@dataclass
class DriftFinding:
    code: str
    severity: str
    statement: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity,
                "statement": self.statement, "evidence": self.evidence}


@dataclass
class DriftReport:
    severity: str
    drifted: bool
    signals: dict[str, Any]
    findings: list[DriftFinding]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "drifted": self.drifted,
                "summary": self.summary, "signals": self.signals,
                "findings": [finding.to_dict() for finding in self.findings]}


def analyze_drift(
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    feature_paths: list[str],
    target_mean: float,
    target_scale: float,
    champion_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    static_metrics: dict[str, Any],
    config: DriftConfig,
) -> DriftReport:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(feature_scale, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(mean) or len(scale) != len(mean):
        raise ValueError("drift statistics do not match the feature matrix")
    if y.shape != (x.shape[0],):
        raise ValueError("drift target length does not match the feature matrix")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("drift detection requires finite inputs and targets")

    champion_rmse = float(champion_metrics["rmse"])
    reference_rmse = float(reference_metrics["rmse"])
    static_rmse = float(static_metrics["rmse"])
    performance_delta = _ratio(champion_rmse - reference_rmse, reference_rmse)
    static_delta = _ratio(champion_rmse - static_rmse, static_rmse)

    feature_z = (x.mean(axis=0) - mean) / np.maximum(scale, 1e-12)
    abs_z = np.abs(feature_z)
    max_abs_z = float(abs_z.max()) if len(abs_z) else 0.0
    fraction_shifted = float(np.mean(abs_z >= config.feature_fraction_z)) if len(abs_z) else 0.0
    scale_ratio = x.std(axis=0) / np.maximum(scale, 1e-12)
    max_scale_deviation = float(np.max(np.abs(scale_ratio - 1.0))) if len(scale_ratio) else 0.0
    order = np.argsort(-abs_z)[:3]
    top_features = [
        {"feature": describe_feature(feature_paths[index]), "mean_z": round(float(feature_z[index]), 6),
         "std_ratio": round(float(scale_ratio[index]), 6)}
        for index in order
    ]

    target_mean_shift = abs(float(y.mean()) - target_mean) / max(abs(target_scale), 1e-12)
    target_std_ratio = float(y.std()) / max(abs(target_scale), 1e-12)
    target_scale_deviation = abs(target_std_ratio - 1.0)

    findings: list[DriftFinding] = []

    perf_rank = 0
    if performance_delta >= config.performance_critical:
        perf_rank = 2
    elif performance_delta >= config.performance_warn:
        perf_rank = 1
    static_rank = 0
    if static_delta >= config.static_critical:
        static_rank = 2
    elif static_delta >= config.static_warn:
        static_rank = 1
    if perf_rank:
        findings.append(DriftFinding(
            code="performance_degradation", severity=_severity(perf_rank),
            statement=(f"验证 RMSE 为 {champion_rmse:.3f}，较 champion 参考值 "
                       f"{reference_rmse:.3f} 上升 {performance_delta * 100:.1f}%"),
            evidence={"champion_rmse": round(champion_rmse, 6),
                      "reference_rmse": round(reference_rmse, 6),
                      "relative_delta": round(performance_delta, 6)},
        ))
    if static_rank:
        findings.append(DriftFinding(
            code="champion_below_static", severity=_severity(static_rank),
            statement=(f"当前 champion 在同一批次上比官方静态基线差 "
                       f"{static_delta * 100:.1f}%（{champion_rmse:.3f} vs {static_rmse:.3f}）"),
            evidence={"champion_rmse": round(champion_rmse, 6),
                      "static_rmse": round(static_rmse, 6),
                      "relative_delta": round(static_delta, 6)},
        ))

    feature_rank = 0
    if max_abs_z >= config.feature_max_z_critical or fraction_shifted >= config.feature_fraction_critical:
        feature_rank = 2
    elif max_abs_z >= config.feature_max_z_warn or fraction_shifted >= config.feature_fraction_warn:
        feature_rank = 1
    if feature_rank:
        top_text = "、".join(item["feature"] for item in top_features)
        findings.append(DriftFinding(
            code="input_distribution_shift", severity=_severity(feature_rank),
            statement=(f"输入特征相对训练统计量发生偏移：最大 |z|={max_abs_z:.2f}，"
                       f"{fraction_shifted * 100:.0f}% 的特征超过 {config.feature_fraction_z:.1f}σ；"
                       f"偏移最大的是 {top_text}"),
            evidence={"max_abs_mean_z": round(max_abs_z, 6),
                      "shifted_feature_fraction": round(fraction_shifted, 6),
                      "max_std_ratio_deviation": round(max_scale_deviation, 6),
                      "top_features": top_features},
        ))

    target_rank = 0
    if target_mean_shift >= config.target_mean_shift_critical:
        target_rank = 2
    elif target_mean_shift >= config.target_mean_shift_warn:
        target_rank = 1
    if target_rank:
        findings.append(DriftFinding(
            code="target_distribution_shift", severity=_severity(target_rank),
            statement=(f"目标均值偏移 {target_mean_shift:.2f} 个目标标准差"
                       f"（批次均值 {float(y.mean()):.2f}，训练均值 {target_mean:.2f}）"),
            evidence={"target_mean": round(float(y.mean()), 6),
                      "reference_target_mean": round(target_mean, 6),
                      "mean_shift_in_target_scales": round(target_mean_shift, 6),
                      "std_ratio": round(target_std_ratio, 6),
                      "scale_deviation": round(target_scale_deviation, 6)},
        ))

    severity = _severity(max([SEVERITY_RANK[finding.severity] for finding in findings] or [0]))
    signals = {
        "batch_rows": int(x.shape[0]),
        "champion_rmse": round(champion_rmse, 6),
        "champion_mae": (round(float(champion_metrics["mae"]), 6)
                         if champion_metrics.get("mae") is not None else None),
        "reference_rmse": round(reference_rmse, 6),
        "static_rmse": round(static_rmse, 6),
        "performance_delta": round(performance_delta, 6),
        "static_delta": round(static_delta, 6),
        "feature_max_abs_mean_z": round(max_abs_z, 6),
        "feature_shifted_fraction": round(fraction_shifted, 6),
        "feature_max_std_ratio_deviation": round(max_scale_deviation, 6),
        "target_mean_shift": round(target_mean_shift, 6),
        "target_std_ratio": round(target_std_ratio, 6),
        "top_features": top_features,
    }
    if severity == "stable":
        summary = f"未检测到显著漂移（RMSE {champion_rmse:.3f}，参考 {reference_rmse:.3f}）。"
    else:
        causes = "；".join(finding.statement for finding in findings)
        summary = f"{'严重' if severity == 'critical' else '警告'}级漂移：{causes}"
    return DriftReport(severity=severity, drifted=severity != "stable",
                       signals=signals, findings=findings, summary=summary)
