"""Harness configuration with strict validation and typo detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from .util import load_json

CONFIG_FORMAT = "harness.config.v1"

DEFAULTS: dict[str, Any] = {
    "format": CONFIG_FORMAT,
    "profile": "C",
    "task": "regression",
    "metric": "rmse",
    "drift": {
        "performance_warn": 0.05,
        "performance_critical": 0.15,
        "static_warn": 0.03,
        "static_critical": 0.10,
        "feature_max_z_warn": 2.0,
        "feature_max_z_critical": 4.0,
        "feature_fraction_z": 1.5,
        "feature_fraction_warn": 0.15,
        "feature_fraction_critical": 0.40,
        "target_mean_shift_warn": 0.25,
        "target_mean_shift_critical": 0.75,
    },
    "experiments": {
        "min_attempts": 3,
        "max_candidates": 3,
        "epochs": 3,
        "exploration_every": 3,
        "time_budget_seconds": 180.0,
        "experiment_timeout_seconds": 120.0,
        "learning_rate_multipliers": [0.5, 1.0, 2.0],
        "replay_rows": 3000,
        "batch_size": None,
    },
    "gate": {
        "min_improvement": 0.01,
        "guard_max_degradation": 0.01,
        "static_tolerance": 0.0,
        "rollback_margin": 0.05,
    },
    "guard": {
        "rows": 2000,
        "seed": 42,
    },
    "logging": {
        "swanlab": "offline",
        "project": "harness-cpu-c",
        "experiment": "cpu-c-harness",
        "logdir": "logs/swanlab",
        "description": "Offline CPU-C drift-aware training harness",
    },
}


@dataclass(frozen=True)
class DriftConfig:
    performance_warn: float
    performance_critical: float
    static_warn: float
    static_critical: float
    feature_max_z_warn: float
    feature_max_z_critical: float
    feature_fraction_z: float
    feature_fraction_warn: float
    feature_fraction_critical: float
    target_mean_shift_warn: float
    target_mean_shift_critical: float


@dataclass(frozen=True)
class ExperimentConfig:
    min_attempts: int
    max_candidates: int
    epochs: int
    exploration_every: int
    time_budget_seconds: float
    experiment_timeout_seconds: float
    learning_rate_multipliers: tuple[float, ...]
    replay_rows: int
    batch_size: int | None


@dataclass(frozen=True)
class GateConfig:
    min_improvement: float
    guard_max_degradation: float
    static_tolerance: float
    rollback_margin: float


@dataclass(frozen=True)
class GuardConfig:
    rows: int
    seed: int


@dataclass(frozen=True)
class LoggingConfig:
    swanlab: str
    project: str
    experiment: str
    logdir: str
    description: str


@dataclass(frozen=True)
class HarnessConfig:
    profile: str
    task: str
    metric: str
    drift: DriftConfig
    experiments: ExperimentConfig
    gate: GateConfig
    guard: GuardConfig
    logging: LoggingConfig
    source: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _merge(base: Any, override: Any, path: str = "") -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        unknown = set(override) - set(base)
        if unknown:
            raise ValueError(f"unknown configuration key(s) at {path or '<root>'}: {sorted(unknown)}")
        return {key: _merge(base[key], override.get(key, base[key]), f"{path}.{key}" if path else key)
                for key in base}
    if isinstance(base, list) and isinstance(override, list):
        return list(override)
    return override


def _number(value: Any, name: str, *, low: float = -math.inf, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return float(value)


def _integer(value: Any, name: str, *, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a nonempty string of at most 200 characters")
    return value


def from_mapping(raw: dict[str, Any], *, source: dict[str, Any] | None = None) -> HarnessConfig:
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a JSON object")
    merged = _merge(DEFAULTS, raw)
    if merged.get("format") != CONFIG_FORMAT:
        raise ValueError(f"unsupported configuration format: {merged.get('format')!r}")
    profile = _text(merged.get("profile"), "profile")
    task = _text(merged.get("task"), "task")
    metric = _text(merged.get("metric"), "metric")
    if profile != "C":
        raise ValueError("this minimal harness build supports baseline profile C only")
    if task != "regression" or metric != "rmse":
        raise ValueError("this minimal harness build supports the CPU-C regression task with RMSE only")

    d = merged["drift"]
    drift = DriftConfig(
        performance_warn=_number(d["performance_warn"], "drift.performance_warn", low=0, high=10),
        performance_critical=_number(d["performance_critical"], "drift.performance_critical", low=0, high=10),
        static_warn=_number(d["static_warn"], "drift.static_warn", low=0, high=10),
        static_critical=_number(d["static_critical"], "drift.static_critical", low=0, high=10),
        feature_max_z_warn=_number(d["feature_max_z_warn"], "drift.feature_max_z_warn", low=0, high=1000),
        feature_max_z_critical=_number(d["feature_max_z_critical"], "drift.feature_max_z_critical", low=0, high=1000),
        feature_fraction_z=_number(d["feature_fraction_z"], "drift.feature_fraction_z", low=0, high=1000),
        feature_fraction_warn=_number(d["feature_fraction_warn"], "drift.feature_fraction_warn", low=0, high=1),
        feature_fraction_critical=_number(d["feature_fraction_critical"], "drift.feature_fraction_critical", low=0, high=1),
        target_mean_shift_warn=_number(d["target_mean_shift_warn"], "drift.target_mean_shift_warn", low=0, high=100),
        target_mean_shift_critical=_number(d["target_mean_shift_critical"], "drift.target_mean_shift_critical", low=0, high=100),
    )
    for prefix in ("performance", "static", "feature_max_z", "feature_fraction", "target_mean_shift"):
        warn = getattr(drift, f"{prefix}_warn")
        critical = getattr(drift, f"{prefix}_critical")
        if critical < warn:
            raise ValueError(f"drift.{prefix}_critical must be >= drift.{prefix}_warn")
    if drift.feature_max_z_warn < drift.feature_fraction_z:
        raise ValueError("drift.feature_max_z_warn must be >= drift.feature_fraction_z")

    e = merged["experiments"]
    multipliers = e["learning_rate_multipliers"]
    if not isinstance(multipliers, list) or not 1 <= len(multipliers) <= 8:
        raise ValueError("experiments.learning_rate_multipliers must be a list of 1..8 numbers")
    multipliers = tuple(_number(value, "experiments.learning_rate_multipliers[]", low=1e-6, high=10)
                        for value in multipliers)
    batch_size = e.get("batch_size")
    if batch_size is not None:
        batch_size = _integer(batch_size, "experiments.batch_size", low=1, high=65536)
    experiments = ExperimentConfig(
        min_attempts=_integer(e["min_attempts"], "experiments.min_attempts", low=0, high=1000),
        max_candidates=_integer(e["max_candidates"], "experiments.max_candidates", low=1, high=16),
        epochs=_integer(e["epochs"], "experiments.epochs", low=1, high=1000),
        exploration_every=_integer(e["exploration_every"], "experiments.exploration_every", low=0, high=10000),
        time_budget_seconds=_number(e["time_budget_seconds"], "experiments.time_budget_seconds", low=0.1, high=86400),
        experiment_timeout_seconds=_number(e["experiment_timeout_seconds"], "experiments.experiment_timeout_seconds",
                                           low=0.1, high=86400),
        learning_rate_multipliers=multipliers,
        replay_rows=_integer(e["replay_rows"], "experiments.replay_rows", low=0, high=2_000_000),
        batch_size=batch_size,
    )

    g = merged["gate"]
    gate = GateConfig(
        min_improvement=_number(g["min_improvement"], "gate.min_improvement", low=0, high=0.9),
        guard_max_degradation=_number(g["guard_max_degradation"], "gate.guard_max_degradation", low=0, high=0.9),
        static_tolerance=_number(g["static_tolerance"], "gate.static_tolerance", low=0, high=0.9),
        rollback_margin=_number(g["rollback_margin"], "gate.rollback_margin", low=0, high=1),
    )

    u = merged["guard"]
    guard = GuardConfig(
        rows=_integer(u["rows"], "guard.rows", low=1, high=200000),
        seed=_integer(u["seed"], "guard.seed", low=0, high=2**32 - 1),
    )

    l = merged["logging"]
    mode = _text(l["swanlab"], "logging.swanlab")
    if mode not in ("off", "offline", "cloud", "required"):
        raise ValueError("logging.swanlab must be one of: off, offline, cloud, required")
    description = l["description"]
    if not isinstance(description, str) or len(description) > 500:
        raise ValueError("logging.description must be a string of at most 500 characters")
    logging = LoggingConfig(
        swanlab=mode,
        project=_text(l["project"], "logging.project"),
        experiment=_text(l["experiment"], "logging.experiment"),
        logdir=_text(l["logdir"], "logging.logdir"),
        description=description,
    )
    return HarnessConfig(profile=profile, task=task, metric=metric, drift=drift, experiments=experiments,
                         gate=gate, guard=guard, logging=logging, source=source or {})


def load_config(path: Path | str | None = None) -> HarnessConfig:
    if path is None:
        raw = dict(DEFAULTS)
        return from_mapping(raw, source={"path": None})
    path = Path(path)
    raw = load_json(path)
    return from_mapping(raw, source={"path": str(path)})


def config_to_dict(config: HarnessConfig) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "profile": config.profile,
        "task": config.task,
        "metric": config.metric,
        "drift": config.drift.__dict__,
        "experiments": {**config.experiments.__dict__,
                        "learning_rate_multipliers": list(config.experiments.learning_rate_multipliers)},
        "gate": config.gate.__dict__,
        "guard": config.guard.__dict__,
        "logging": config.logging.__dict__,
    }
