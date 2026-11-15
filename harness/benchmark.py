"""Aggregate a drift-stream run into comparable benchmark statistics.

The benchmark compares, for every batch, the model selected by the Harness
with the untouched official static checkpoint evaluated on the same rows.
Positive gain means the selected model has lower RMSE than the static floor.
"""

from __future__ import annotations

import statistics
from typing import Any


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metrics(group) -> dict[str, float | None]:
    if not isinstance(group, dict):
        return {"rmse": None, "mae": None, "nasa_score": None}
    return {name: _number(group.get(name)) for name in ("rmse", "mae", "nasa_score")}


def batch_result(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    static = _metrics((metrics.get("static") or {}).get("batch"))
    before = _metrics((metrics.get("champion_before") or {}).get("batch"))
    selected = _metrics(((record.get("selected_model") or {}).get("metrics") or {}).get("batch"))
    static_guard = _metrics((metrics.get("static") or {}).get("guard"))
    selected_guard = _metrics(((record.get("selected_model") or {}).get("metrics") or {}).get("guard"))
    if record.get("error"):
        selected = {"rmse": None, "mae": None, "nasa_score": None}
        selected_guard = {"rmse": None, "mae": None, "nasa_score": None}
    static_rmse, selected_rmse = static["rmse"], selected["rmse"]
    gain_abs = (static_rmse - selected_rmse
                if static_rmse is not None and selected_rmse is not None else None)
    gain_rel = (gain_abs / static_rmse
                if gain_abs is not None and static_rmse else None)
    guard_gain_abs = ((static_guard["rmse"] - selected_guard["rmse"])
                      if static_guard["rmse"] is not None and selected_guard["rmse"] is not None else None)
    experiments = [item for item in record.get("experiments") or [] if isinstance(item, dict)]
    training_seconds = sum(
        float((item.get("training") or {}).get("training_seconds") or 0.0)
        for item in experiments)
    gate_passes = sum(
        1 for item in experiments
        if (item.get("gate") or {}).get("passed"))
    metadata = (record.get("batch") or {}).get("metadata") or {}
    return {
        "batch_id": (record.get("batch") or {}).get("id"),
        "batch_index": (record.get("batch") or {}).get("index"),
        "drift_type": metadata.get("drift_type"),
        "drift_severity": (record.get("drift") or {}).get("severity"),
        "decision": record.get("decision"),
        "static_rmse": static_rmse,
        "static_mae": static["mae"],
        "static_nasa_score": static["nasa_score"],
        "champion_before_rmse": before["rmse"],
        "selected_rmse": selected_rmse,
        "selected_mae": selected["mae"],
        "selected_nasa_score": selected["nasa_score"],
        "static_guard_rmse": static_guard["rmse"],
        "selected_guard_rmse": selected_guard["rmse"],
        "gain_abs": gain_abs,
        "gain_rel": gain_rel,
        "guard_gain_abs": guard_gain_abs,
        "experiments": len(experiments),
        "gate_passes": gate_passes,
        "training_seconds": training_seconds,
        "reason": record.get("reason"),
        "error": record.get("error"),
    }


def summarize(rows: list[dict[str, Any]], *, wall_seconds: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("benchmark needs at least one batch result")
    pairs = [(row["static_rmse"], row["selected_rmse"]) for row in rows
             if row["static_rmse"] is not None and row["selected_rmse"] is not None]
    if not pairs:
        raise ValueError("benchmark rows have no comparable RMSE values")
    gains_rel = [((static - selected) / static) for static, selected in pairs if static]
    gains_abs = [static - selected for static, selected in pairs]
    improved = sum(1 for value in gains_abs if value > 1e-9)
    worse = sum(1 for value in gains_abs if value < -1e-9)
    equal = len(gains_abs) - improved - worse
    decisions = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    drifts = {}
    for row in rows:
        key = row.get("drift_severity") or "none"
        drifts[key] = drifts.get(key, 0) + 1
    attempts = sum(int(row.get("experiments") or 0) for row in rows)
    training_seconds = sum(float(row.get("training_seconds") or 0.0) for row in rows)
    return {
        "batches": len(rows),
        "comparable_batches": len(pairs),
        "attempts": attempts,
        "gate_passes": sum(int(row.get("gate_passes") or 0) for row in rows),
        "decisions": decisions,
        "drift_severities": drifts,
        "improved_batches": improved,
        "equal_batches": equal,
        "worse_batches": worse,
        "static_floor_violations": worse,
        "mean_static_rmse": statistics.mean(static for static, _ in pairs),
        "median_static_rmse": statistics.median(static for static, _ in pairs),
        "mean_selected_rmse": statistics.mean(selected for _, selected in pairs),
        "median_selected_rmse": statistics.median(selected for _, selected in pairs),
        "mean_gain_rel": statistics.mean(gains_rel) if gains_rel else 0.0,
        "median_gain_rel": statistics.median(gains_rel) if gains_rel else 0.0,
        "best_gain_rel": max(gains_rel) if gains_rel else 0.0,
        "worst_gain_rel": min(gains_rel) if gains_rel else 0.0,
        "net_rmse_gain": sum(gains_abs),
        "cumulative_gain": sum(max(0.0, value) for value in gains_abs),
        "cumulative_regret": sum(max(0.0, -value) for value in gains_abs),
        "gain_per_attempt": (sum(gains_abs) / attempts) if attempts else 0.0,
        "training_seconds": training_seconds,
        "wall_seconds": wall_seconds,
        "attempts_per_batch": attempts / len(rows),
    }
