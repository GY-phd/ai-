"""Turn a harness decision record into SwanLab metrics and text records.

Kept free of SwanLab imports so the payload can be unit-tested and logged by
any compatible backend.  Every value is converted to a finite number; the
decision file remains the authoritative machine-readable record.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SEVERITY_RANK = {"stable": 0, "warning": 1, "critical": 2}


def _key_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return text[:64] or "item"


def _clean(values: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            result[key] = int(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                result[key] = number
    return result


def _metric_group(prefix: str, metrics: dict[str, Any] | None) -> dict[str, float | int]:
    if not isinstance(metrics, dict):
        return {}
    values = {}
    for name in ("rmse", "mae", "nasa_score", "bias", "n"):
        if metrics.get(name) is not None:
            values[f"{prefix}_{name}"] = metrics[name]
    return _clean(values)


def batch_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    metrics = record.get("metrics") or {}
    drift = record.get("drift") or {}
    signals = drift.get("signals") or {}
    counters = record.get("counters") or {}
    budget = record.get("budget") or {}
    batch_info = record.get("batch") or {}
    training = record.get("training_data") or {}
    decision = record.get("decision")
    values: dict[str, Any] = {
        "batch/index": batch_info.get("index"),
        "batch/decision_promote": decision == "promote",
        "batch/decision_hold": decision == "hold",
        "batch/decision_rollback": decision == "rollback",
        "batch/dry_run": bool(record.get("dry_run")),
        "batch/drift_severity": SEVERITY_RANK.get(drift.get("severity"), -1),
        "batch/experiments_planned": len(record.get("experiments_planned") or []),
        "batch/experiments_run": len(record.get("experiments") or []),
        "batch/attempts_total": counters.get("attempts"),
        "batch/successful_experiments": counters.get("successful_experiments"),
        "batch/failed_experiments": counters.get("failed_experiments"),
        "batch/promotions": counters.get("promotions"),
        "batch/rollbacks": counters.get("rollbacks"),
        "batch/holds": counters.get("holds"),
        "batch/errors": counters.get("errors"),
        "batch/time_budget_seconds": budget.get("time_budget_seconds"),
        "batch/experiment_seconds": budget.get("elapsed_seconds"),
        "batch/rows": ((metrics.get("champion_before") or {}).get("batch") or {}).get("n"),
        "batch/guard_rows": ((metrics.get("champion_before") or {}).get("guard") or {}).get("n"),
        "batch/train_rows": training.get("rows"),
        "batch/excluded_ids": training.get("excluded_ids"),
    }
    for label, group in (("champion", metrics.get("champion_before")),
                         ("static", metrics.get("static"))):
        if isinstance(group, dict):
            values.update(_metric_group(f"batch/{label}_batch", group.get("batch")))
            values.update(_metric_group(f"batch/{label}_guard", group.get("guard")))
    selected = (record.get("selected_model") or {}).get("metrics") or {}
    values.update(_metric_group("batch/selected_batch", selected.get("batch")))
    values.update(_metric_group("batch/selected_guard", selected.get("guard")))
    for name, value in signals.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"drift/{name}"] = value
    return _clean(values)


def candidate_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    values: dict[str, Any] = {}
    for index, experiment in enumerate(record.get("experiments") or [], 1):
        if not isinstance(experiment, dict):
            continue
        prefix = f"candidates/c{index:02d}"
        spec = experiment.get("spec") or {}
        metadata = experiment.get("metrics") or {}
        gate = experiment.get("gate") or {}
        training = experiment.get("training") or {}
        values.update(_metric_group(f"{prefix}/batch", metadata.get("batch")))
        values.update(_metric_group(f"{prefix}/guard", metadata.get("guard")))
        values.update({
            f"{prefix}/gate_passed": bool(gate.get("passed")),
            f"{prefix}/gate_failed_checks": len(gate.get("failed_checks") or []),
            f"{prefix}/improvement": gate.get("improvement"),
            f"{prefix}/epochs": spec.get("epochs"),
            f"{prefix}/learning_rate": spec.get("learning_rate"),
            f"{prefix}/batch_size": spec.get("batch_size"),
            f"{prefix}/training_seconds": training.get("training_seconds"),
            f"{prefix}/duration_seconds": training.get("duration_seconds"),
            f"{prefix}/initial_steps": training.get("initial_steps"),
            f"{prefix}/final_steps": training.get("final_steps"),
            f"{prefix}/report_mismatch": experiment.get("report_mismatch"),
            f"{prefix}/has_error": bool(experiment.get("error")),
            f"{prefix}/parameter_count": (experiment.get("model_meta") or {}).get("parameter_count"),
        })
    return _clean(values)


def epoch_metrics(record: dict[str, Any]) -> dict[int, dict[str, float | int]]:
    """Return ``epoch -> metrics`` with one curve set per batch/candidate."""
    by_epoch: dict[int, dict[str, float | int]] = {}
    batch_index = int((record.get("batch") or {}).get("index") or 0)
    for index, experiment in enumerate(record.get("experiments") or [], 1):
        if not isinstance(experiment, dict):
            continue
        prefix = f"curves/b{batch_index:04d}/c{index:02d}"
        for item in experiment.get("history") or []:
            if not isinstance(item, dict):
                continue
            epoch = item.get("epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                continue
            validation = item.get("validation") or {}
            values = _clean({
                f"{prefix}/train_loss": item.get("loss"),
                f"{prefix}/val_rmse": validation.get("rmse"),
                f"{prefix}/val_mae": validation.get("mae"),
            })
            if values:
                by_epoch.setdefault(epoch, {}).update(values)
    return by_epoch


def text_records(record: dict[str, Any]) -> list[tuple[str, str]]:
    batch = record.get("batch") or {}
    batch_id = _key_part(str(batch.get("id") or "batch"))
    drift = record.get("drift") or {}
    selected = record.get("selected_model") or {}
    lines = [
        f"batch: {batch.get('id')} (index {batch.get('index')})",
        f"decision: {record.get('decision')}",
        f"selected_model: {selected.get('stage_id')} (generation {selected.get('generation')})",
        f"reason: {record.get('reason')}",
        f"drift: {drift.get('severity')} - {drift.get('summary')}",
    ]
    for hypothesis in record.get("hypotheses") or []:
        lines.append(f"hypothesis[{hypothesis.get('code')}]: {hypothesis.get('statement')}")
    for experiment in record.get("experiments") or []:
        gate = experiment.get("gate") or {}
        lines.append(
            f"experiment[{experiment.get('stage_id')}]: batch_rmse="
            f"{((experiment.get('metrics') or {}).get('batch') or {}).get('rmse')} "
            f"gate_passed={gate.get('passed')} failed={gate.get('failed_checks')} "
            f"error={experiment.get('error')}")
    if record.get("error"):
        lines.append(f"error: {record.get('error')}")
    texts = [
        (f"drift/{batch_id}", f"{drift.get('summary')}\n" + json.dumps(
            drift.get("findings") or [], ensure_ascii=False, allow_nan=False)),
        (f"decision/{batch_id}", "\n".join(lines)),
    ]
    return texts


def final_metrics(status: dict[str, Any]) -> dict[str, float | int]:
    champion = status.get("champion") or {}
    metrics = champion.get("metrics") or {}
    batch = metrics.get("batch") or {}
    guard = metrics.get("guard") or {}
    counters = status.get("counters") or {}
    values: dict[str, Any] = {
        "final/champion_generation": champion.get("generation"),
        "final/champion_is_static": bool(champion.get("is_static")),
        "final/champion_batch_rmse": batch.get("rmse"),
        "final/champion_batch_mae": batch.get("mae"),
        "final/champion_guard_rmse": guard.get("rmse"),
        "final/champion_guard_mae": guard.get("mae"),
        "final/batches": len(status.get("batches") or []),
    }
    for name, value in counters.items():
        values[f"final/{name}"] = value
    return _clean(values)
