"""Diagnosis, experiment planning, promotion gates and rollback policy.

The policy is deliberately deterministic and budget-aware.  It never promotes
merely because training ran: a candidate must improve on the batch validation
set, must not regress the fixed public guard set, and may not fall below the
official static checkpoint on either set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .config import ExperimentConfig, GateConfig
from .drift import DriftReport


@dataclass
class Hypothesis:
    code: str
    statement: str
    action: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "statement": self.statement,
                "action": self.action, "priority": self.priority}


@dataclass
class ExperimentSpec:
    index: int
    epochs: int
    learning_rate: float
    learning_rate_multiplier: float
    batch_size: int
    hypothesis: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "learning_rate_multiplier": self.learning_rate_multiplier,
                "batch_size": self.batch_size, "hypothesis": self.hypothesis,
                "reason": self.reason}


@dataclass
class GateResult:
    passed: bool
    improvement: float
    failed_checks: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "improvement": self.improvement,
                "failed_checks": list(self.failed_checks), "checks": self.checks}


@dataclass
class RollbackDecision:
    should_rollback: bool
    target: str | None
    reason: str
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"should_rollback": self.should_rollback, "target": self.target,
                "reason": self.reason, "checks": self.checks}


def diagnose(report: DriftReport, *, attempts: int, batch_index: int,
             config: ExperimentConfig) -> list[Hypothesis]:
    codes = {finding.code for finding in report.findings}
    hypotheses: list[Hypothesis] = []
    if "target_distribution_shift" in codes:
        hypotheses.append(Hypothesis(
            "concept_drift", "目标分布发生偏移，冻结的输入标准化与输出缩放可能失配；应通过受控续训重新校准输出层。",
            "adapt_weights", 100))
    if "input_distribution_shift" in codes:
        hypotheses.append(Hypothesis(
            "covariate_shift", "输入传感器分布发生偏移，模型看到的标准化空间与训练时不同；应用较小学习率、短轮次适配权重。",
            "adapt_weights", 90))
    if "performance_degradation" in codes and not ({"target_distribution_shift", "input_distribution_shift"} & codes):
        hypotheses.append(Hypothesis(
            "model_staleness", "输入与目标分布基本稳定但误差上升，模型可能已过时；应以基准学习率做一次短程续训。",
            "adapt_weights", 80))
    if "champion_below_static" in codes:
        hypotheses.append(Hypothesis(
            "champion_regression", "当前 champion 已劣于官方静态基线；若本次实验无法恢复，应回滚到上一稳定检查点或静态基线。",
            "prepare_rollback", 70))
    if not hypotheses:
        remaining = max(0, config.min_attempts - attempts)
        if remaining:
            hypotheses.append(Hypothesis(
                "exploration", f"未检测到漂移，但仍需完成至少 {config.min_attempts} 次尝试；本次做一次低成本续训以保留探索证据。",
                "explore", 50))
        elif config.exploration_every and batch_index % config.exploration_every == 0:
            hypotheses.append(Hypothesis(
                "exploration", "未检测到漂移，按周期性探索计划做一次低成本实验，避免长期只监控不实验。",
                "explore", 40))
        else:
            hypotheses.append(Hypothesis(
                "monitor_only", "未检测到漂移且尝试配额已满足；本批次保持当前 champion，不消耗训练资源。",
                "monitor", 0))
    hypotheses.sort(key=lambda item: -item.priority)
    return hypotheses


def _ordered_multipliers(config: ExperimentConfig, preferred: list[float]) -> list[float]:
    def rank(value: float) -> tuple[int, float]:
        distances = [(abs(value - target), index) for index, target in enumerate(preferred)]
        index = min(distances)[1]
        return (index, value)

    return sorted(config.learning_rate_multipliers, key=rank)


def plan_experiments(report: DriftReport, hypotheses: list[Hypothesis], *,
                     champion_meta: dict[str, Any], attempts: int, batch_index: int,
                     config: ExperimentConfig) -> list[ExperimentSpec]:
    if report.severity == "critical":
        count = config.max_candidates
    elif report.severity == "warning":
        count = min(config.max_candidates, max(2, config.min_attempts - attempts))
    elif attempts < config.min_attempts:
        count = min(config.max_candidates, config.min_attempts - attempts)
    elif config.exploration_every and batch_index % config.exploration_every == 0:
        count = 1
    else:
        count = 0
    if count <= 0:
        return []

    primary = hypotheses[0].code if hypotheses else "exploration"
    if primary == "concept_drift":
        multipliers = _ordered_multipliers(config, [2.0, 1.0, 0.5])
        epoch_factor = 1.5
    elif primary == "covariate_shift":
        multipliers = _ordered_multipliers(config, [0.5, 1.0, 2.0])
        epoch_factor = 1.0
    elif primary in ("model_staleness", "champion_regression"):
        multipliers = _ordered_multipliers(config, [1.0, 2.0, 0.5])
        epoch_factor = 1.0
    else:
        multipliers = _ordered_multipliers(config, [1.0, 0.5, 2.0])
        epoch_factor = 1.0

    champion_lr = float(champion_meta["learning_rate"])
    champion_batch = int(champion_meta["batch_size"])
    batch_size = int(config.batch_size) if config.batch_size is not None else champion_batch
    epochs = max(1, min(1000, int(math.ceil(config.epochs * epoch_factor))))

    specs: list[ExperimentSpec] = []
    seen: set[tuple[float, int, int]] = set()
    cycle = 0
    # A short multiplier list must not silently cap min_attempts or
    # max_candidates.  Later cycles reuse the multipliers with one extra epoch,
    # preserving distinct (learning rate, epochs, batch size) experiments.
    while len(specs) < count and cycle <= count:
        for multiplier in multipliers:
            if len(specs) >= count:
                break
            learning_rate = min(1.0, max(1e-9, champion_lr * float(multiplier)))
            candidate_epochs = max(1, min(1000, epochs + cycle))
            key = (round(learning_rate, 12), candidate_epochs, batch_size)
            if key in seen:
                continue
            seen.add(key)
            specs.append(ExperimentSpec(
                index=len(specs) + 1, epochs=candidate_epochs, learning_rate=learning_rate,
                learning_rate_multiplier=float(multiplier), batch_size=batch_size,
                hypothesis=primary,
                reason=(f"{primary}: 从 champion 的 {champion_lr:g} 学习率乘以 "
                        f"{multiplier:g}，训练 {candidate_epochs} 个 epoch"),
            ))
        cycle += 1
    return specs


def gate_candidate(*, candidate: dict[str, dict[str, Any]],
                   champion: dict[str, dict[str, Any]],
                   static: dict[str, dict[str, Any]],
                   config: GateConfig) -> GateResult:
    candidate_batch = float(candidate["batch"]["rmse"])
    candidate_guard = float(candidate["guard"]["rmse"])
    champion_batch = float(champion["batch"]["rmse"])
    champion_guard = float(champion["guard"]["rmse"])
    static_batch = float(static["batch"]["rmse"])
    static_guard = float(static["guard"]["rmse"])
    values = (candidate_batch, candidate_guard, champion_batch, champion_guard,
              static_batch, static_guard)
    finite = all(math.isfinite(value) and value >= 0 for value in values)

    improvement = ((champion_batch - candidate_batch) / max(champion_batch, 1e-12)
                   if finite else 0.0)
    checks: dict[str, Any] = {
        "finite_metrics": {"passed": finite, "value": list(values)},
        "beats_champion_on_batch": {
            "passed": finite and improvement >= config.min_improvement,
            "value": improvement if finite else None,
            "required": config.min_improvement,
        },
        "not_worse_than_static_on_batch": {
            "passed": finite and candidate_batch <= static_batch * (1.0 + config.static_tolerance),
            "value": candidate_batch if finite else None,
            "limit": static_batch * (1.0 + config.static_tolerance),
        },
        "guard_not_degraded": {
            "passed": finite and candidate_guard <= champion_guard * (1.0 + config.guard_max_degradation),
            "value": candidate_guard if finite else None,
            "limit": champion_guard * (1.0 + config.guard_max_degradation),
        },
        "guard_not_worse_than_static": {
            "passed": finite and candidate_guard <= static_guard * (1.0 + config.static_tolerance),
            "value": candidate_guard if finite else None,
            "limit": static_guard * (1.0 + config.static_tolerance),
        },
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return GateResult(passed=not failed, improvement=improvement,
                      failed_checks=failed, checks=checks)


def best_passing_candidate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [result for result in results if result.get("gate") and result["gate"].passed]
    if not passed:
        return None
    return min(passed, key=lambda result: (
        float(result["metrics"]["batch"]["rmse"]),
        float(result["metrics"]["guard"]["rmse"]),
        float(result.get("training", {}).get("duration_seconds", 0.0)),
    ))


def rollback_decision(*, champion: dict[str, Any], champion_metrics: dict[str, dict[str, Any]],
                      static_metrics: dict[str, dict[str, Any]],
                      previous_metrics: dict[str, dict[str, Any]] | None,
                      previous_stage_id: str | None, config: GateConfig) -> RollbackDecision:
    champion_is_static = bool(champion.get("is_static")) or champion.get("stage_id") == "official-baseline"
    champion_batch = float(champion_metrics["batch"]["rmse"])
    champion_guard = float(champion_metrics["guard"]["rmse"])
    static_batch = float(static_metrics["batch"]["rmse"])
    static_guard = float(static_metrics["guard"]["rmse"])
    batch_delta = (champion_batch - static_batch) / max(static_batch, 1e-12)
    guard_delta = (champion_guard - static_guard) / max(static_guard, 1e-12)
    checks = {
        "champion_is_static": champion_is_static,
        "batch_delta_vs_static": round(batch_delta, 6),
        "guard_delta_vs_static": round(guard_delta, 6),
        "rollback_margin": config.rollback_margin,
    }
    if champion_is_static:
        return RollbackDecision(False, None, "champion 已经是官方静态基线，无法再回滚。", checks)
    if max(batch_delta, guard_delta) < config.rollback_margin:
        return RollbackDecision(
            False, None,
            f"champion 相对静态基线的最大劣化 {max(batch_delta, guard_delta) * 100:.1f}% "
            f"未达到回滚阈值 {config.rollback_margin * 100:.1f}%。", checks)
    if previous_metrics is not None and previous_stage_id:
        previous_batch = float(previous_metrics["batch"]["rmse"])
        previous_guard = float(previous_metrics["guard"]["rmse"])
        previous_improvement = (champion_batch - previous_batch) / max(champion_batch, 1e-12)
        previous_batch_safe = previous_batch <= static_batch * (1.0 + config.static_tolerance)
        previous_guard_safe = previous_guard <= static_guard * (1.0 + config.static_tolerance)
        checks.update({
            "previous_stage_id": previous_stage_id,
            "previous_batch_rmse": round(previous_batch, 6),
            "previous_guard_rmse": round(previous_guard, 6),
            "previous_improvement_vs_champion": round(previous_improvement, 6),
            "previous_batch_not_worse_than_static": previous_batch_safe,
            "previous_guard_not_worse_than_static": previous_guard_safe,
        })
        if previous_improvement >= config.min_improvement and previous_batch_safe and previous_guard_safe:
            return RollbackDecision(
                True, "previous",
                f"champion 劣于静态基线且上一检查点在本批次上好 {previous_improvement * 100:.1f}%，回滚到上一稳定检查点。",
                checks)
    return RollbackDecision(
        True, "static",
        f"champion 相对静态基线劣化 {max(batch_delta, guard_delta) * 100:.1f}%，且没有可用的更优上一检查点，回滚到官方静态基线。",
        checks)
