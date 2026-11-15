"""Adapter around the official CPU baseline package.

The harness never modifies the official baseline.  Training runs in a child
process through the package's public ``python -m baselines.train`` entry point,
which keeps the baseline's isolation rules (new output directory, parent
checkpoint binding, no hidden-file access) intact.  Evaluation re-loads the
checkpoint and recomputes metrics independently; self-reported training logs
are treated as evidence, not as the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .util import load_json, parse_json, read_text, sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_ROOT = REPO_ROOT / "cpu-c-guide-20261110-v1"
BASELINE_CHECKPOINT = "checkpoint.npz"
BASELINE_TRAIN = "public/train.jsonl"
BASELINE_VALIDATION = "public/validation.jsonl"


def default_baseline_root() -> Path:
    return DEFAULT_BASELINE_ROOT


@dataclass
class TrainResult:
    ok: bool
    stage_id: str
    output_dir: str
    checkpoint: str | None
    checkpoint_sha256: str | None
    report: dict[str, Any] | None
    duration_seconds: float
    error: str | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage_id": self.stage_id,
            "output_dir": self.output_dir,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def regression_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if predicted.shape != expected.shape or not len(predicted):
        raise ValueError("prediction and target lengths differ")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(expected)):
        raise ValueError("prediction metrics require finite values")
    error = predicted - expected
    # NASA asymmetric score: over-predicting remaining life is penalised harder.
    penalty = np.where(error < 0, np.expm1(np.clip(-error / 13.0, 0.0, 80.0)),
                       np.expm1(np.clip(error / 10.0, 0.0, 80.0)))
    result = {
        "n": int(len(expected)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score": float(penalty.sum()),
        "bias": float(np.mean(error)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("regression metrics overflowed to a non-finite value")
    return result


def read_history(output_dir: Path | str) -> list[dict[str, Any]]:
    """Read the baseline trainer's per-epoch history for SwanLab curves."""
    path = Path(output_dir) / "history.jsonl"
    if path.is_symlink() or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(read_text(path, 64 * 1024 * 1024).splitlines(), 1):
        if not line.strip():
            continue
        row = parse_json(line)
        if not isinstance(row, dict):
            raise ValueError(f"history line {number} is not an object")
        epoch, step = row.get("epoch"), row.get("step")
        loss, validation = row.get("loss"), row.get("validation")
        if (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
                or isinstance(step, bool) or not isinstance(step, int) or step < 0
                or not isinstance(validation, dict)):
            raise ValueError(f"history line {number} has invalid epoch/step/validation")
        try:
            loss_value = float(loss)
            rmse = float(validation["rmse"])
            mae = float(validation["mae"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"history line {number} has invalid metrics") from exc
        if not all(math.isfinite(value) for value in (loss_value, rmse, mae)):
            raise ValueError(f"history line {number} has non-finite metrics")
        record = {"epoch": epoch, "step": step, "loss": loss_value,
                  "validation": {"rmse": rmse, "mae": mae}}
        if isinstance(row.get("stage_id"), str):
            record["stage_id"] = row["stage_id"]
        if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool):
            record["seed"] = row["seed"]
        records.append(record)
        if len(records) > 10000:
            raise ValueError("history exceeds 10,000 epochs")
    return records


class Baseline:
    """Thin, audited façade over one official baseline directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.profile_config = self.root / "baselines" / "configs" / "C.json"
        self.checkpoint = self.root / BASELINE_CHECKPOINT
        self.public_train = self.root / BASELINE_TRAIN
        self.public_validation = self.root / BASELINE_VALIDATION
        self._modules: dict[str, Any] | None = None
        self._profile: dict[str, Any] | None = None

    # ------------------------------------------------------------------ setup
    def validate(self, expected_profile: str = "C", expected_task: str = "regression") -> dict[str, Any]:
        required = [self.profile_config, self.checkpoint, self.public_train,
                    self.public_validation, self.root / "baselines" / "train.py"]
        for path in required:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"official baseline file is missing or unsafe: {path}")
        profile = load_json(self.profile_config)
        if profile.get("profile") != expected_profile or profile.get("task") != expected_task:
            raise ValueError("official baseline profile does not match the harness configuration")
        if profile.get("task") != "regression":
            raise ValueError("only the regression profile is supported by this build")
        info_path = self.root / "baseline-info.json"
        if info_path.is_file() and not info_path.is_symlink():
            info = load_json(info_path)
            recorded = info.get("checkpoint_sha256")
            actual = sha256_file(self.checkpoint)
            if recorded and recorded != actual:
                raise ValueError("official checkpoint digest does not match baseline-info.json")
        self._profile = profile
        return profile

    def load_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules
        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from baselines.data import flatten, read_dataset  # type: ignore
        from baselines.model import Model  # type: ignore
        self._modules = {"Model": Model, "read_dataset": read_dataset, "flatten": flatten}
        return self._modules

    @property
    def profile(self) -> dict[str, Any]:
        if self._profile is None:
            self.validate()
        assert self._profile is not None
        return self._profile

    # ------------------------------------------------------------------ data
    def load_dataset(self, path: Path | str, expected_paths: list[str] | None = None):
        modules = self.load_modules()
        return modules["read_dataset"](path, self.profile["task"], expected_paths=expected_paths)

    def evaluate(self, checkpoint: Path | str, data: dict[str, Any]) -> dict[str, Any]:
        modules = self.load_modules()
        model = modules["Model"].load(checkpoint)
        if list(model.meta["paths"]) != list(data["paths"]):
            raise ValueError("checkpoint feature schema does not match the evaluation data")
        return regression_metrics(model.predict(data["x"]), data["y"])

    def evaluate_model(self, model, data: dict[str, Any]) -> dict[str, Any]:
        if list(model.meta["paths"]) != list(data["paths"]):
            raise ValueError("checkpoint feature schema does not match the evaluation data")
        return regression_metrics(model.predict(data["x"]), data["y"])

    def load_model(self, checkpoint: Path | str):
        modules = self.load_modules()
        return modules["Model"].load(checkpoint)

    # --------------------------------------------------------------- training
    def train_candidate(
        self,
        *,
        train_path: Path,
        validation_path: Path,
        validation_sha256: str,
        resume_checkpoint: Path,
        output_dir: Path,
        stage_id: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        threads: int,
        timeout_seconds: float,
    ) -> TrainResult:
        output_dir = Path(output_dir)
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("candidate output directory must not exist")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "baselines.train",
            "--profile", self.profile["profile"],
            "--train", str(Path(train_path).resolve()),
            "--validation", str(Path(validation_path).resolve()),
            "--resume", str(Path(resume_checkpoint).resolve()),
            "--output", str(output_dir.resolve()),
            "--stage-id", stage_id,
            "--epochs", str(epochs),
            "--learning-rate", format(float(learning_rate), ".12g"),
            "--batch-size", str(batch_size),
        ]
        environment = os.environ.copy()
        environment["BASELINE_CPU_THREADS"] = str(max(1, min(8, int(threads))))
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, cwd=str(self.root), env=environment, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=max(1.0, float(timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            return TrainResult(
                ok=False, stage_id=stage_id, output_dir=str(output_dir), checkpoint=None,
                checkpoint_sha256=None, report=None, duration_seconds=elapsed,
                error=f"candidate timed out after {timeout_seconds:.1f}s", timed_out=True,
                stdout_tail=_tail(exc.stdout), stderr_tail=_tail(exc.stderr),
            )
        duration = time.monotonic() - started
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        if completed.returncode != 0:
            return TrainResult(
                ok=False, stage_id=stage_id, output_dir=str(output_dir), checkpoint=None,
                checkpoint_sha256=None, report=None, duration_seconds=duration,
                error=f"baseline trainer exited with code {completed.returncode}",
                stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
            )
        try:
            report = load_json(output_dir / "training.json")
            checkpoint = output_dir / "checkpoint.npz"
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise ValueError("candidate checkpoint is missing")
            actual_parent = sha256_file(resume_checkpoint)
            actual_candidate = sha256_file(checkpoint)
            checks = {
                "profile": report.get("profile") == self.profile["profile"],
                "parent": report.get("parent_checkpoint_sha256") == actual_parent,
                "checkpoint": report.get("checkpoint_sha256") == actual_candidate,
                "validation": report.get("validation_sha256") == validation_sha256,
                "steps": isinstance(report.get("final_steps"), int) and isinstance(report.get("initial_steps"), int)
                         and report["final_steps"] > report["initial_steps"],
                "metrics": isinstance(report.get("validation_after"), dict)
                           and math.isfinite(float(report["validation_after"].get("rmse", float("nan")))),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError("candidate report failed integrity checks: " + ", ".join(failed))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed experiment, never swallowed
            return TrainResult(
                ok=False, stage_id=stage_id, output_dir=str(output_dir), checkpoint=None,
                checkpoint_sha256=None, report=None, duration_seconds=duration,
                error=f"candidate verification failed: {exc}",
                stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
            )
        return TrainResult(
            ok=True, stage_id=stage_id, output_dir=str(output_dir), checkpoint=str(checkpoint),
            checkpoint_sha256=actual_candidate, report=report, duration_seconds=duration,
            stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
        )


def _tail(text, limit: int = 4000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-limit:]
