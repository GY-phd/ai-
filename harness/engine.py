"""Autonomous batch loop: detect -> diagnose -> experiment -> gate -> decide.

The engine consumes batch directories in order.  For every batch it writes a
decision file and updates ``state.json`` *before* the next batch is read.  The
champion model is always an on-disk checkpoint under ``workdir/models``; the
official baseline remains untouched and serves as the static floor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

from . import __version__
from .baseline import Baseline, default_baseline_root, read_history
from .config import HarnessConfig, config_to_dict
from .data import guard_dataset, load_optional_batch_metadata, merge_training_sources, replay_dataset
from .drift import analyze_drift
from .journal import Journal
from .policy import (best_passing_candidate, diagnose, gate_candidate, plan_experiments,
                     rollback_decision)
from .state import (add_event, champion_summary, load_state, new_state, processed_batches,
                    record_batch, save_state, STATE_FILE)
from .swanlab_logger import SwanLabLogger
from .telemetry import batch_metrics, candidate_metrics, epoch_metrics, final_metrics, text_records
from .util import (now_utc, relative_to_workdir, resolve_workdir_path,
                   safe_name, sha256_file, unique_path, write_json_atomic)


class HarnessEngine:
    def __init__(self, *, workdir: Path | str, config: HarnessConfig,
                 baseline_root: Path | str | None = None, dry_run: bool = False,
                 force: bool = False, verbose: bool = True):
        self.config = config
        self.workdir = Path(workdir).resolve()
        self.dry_run = dry_run
        self.force = force
        self.verbose = verbose
        self.baseline = Baseline(baseline_root or default_baseline_root())
        self.baseline.validate(config.profile, config.task)
        if self.workdir == self.baseline.root or self.baseline.root in self.workdir.parents:
            raise ValueError("workdir must not be inside the official baseline package")
        self.models_dir = self.workdir / "models"
        self.data_dir = self.workdir / "data"
        self.decisions_dir = self.workdir / "decisions"
        for directory in (self.models_dir, self.data_dir, self.decisions_dir, self.workdir / "logs"):
            directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.workdir / STATE_FILE
        self.journal = Journal(self.workdir / "journal.jsonl")
        self.run_config = {
            "harness_version": __version__,
            "config": config_to_dict(config),
            "baseline": {
                "root": str(self.baseline.root),
                "profile": config.profile,
                "checkpoint_sha256": sha256_file(self.baseline.checkpoint),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "cpu_threads": int(os.environ.get("BASELINE_CPU_THREADS", "1")),
                "gpu_used": False,
                "swanlab_mode": config.logging.swanlab,
            },
        }
        self.swanlab = SwanLabLogger(config.logging, workdir=self.workdir,
                                     run_config=self.run_config)
        self._static_model_cache = None
        self._guard_cache = None
        self.state = self._load_or_bootstrap()

    # ------------------------------------------------------------------ bootstrap
    def _load_or_bootstrap(self) -> dict[str, Any]:
        official_sha = sha256_file(self.baseline.checkpoint)
        if self.state_path.exists():
            state = load_state(self.state_path)
            recorded = state["baseline"].get("official_checkpoint_sha256")
            if recorded and recorded != official_sha:
                raise ValueError("the official baseline checkpoint changed after this workdir was initialised")
            if state["baseline"].get("profile") != self.config.profile:
                raise ValueError("workdir profile does not match the harness configuration")
            current = self._config_dict()
            if state.get("config") != current:
                self.journal.append("config_changed", previous=state.get("config"), current=current)
                add_event(state, "config_changed", previous=state.get("config"), current=current)
                state["config"] = current
            guard = state["data"].get("guard")
            if not isinstance(guard, dict) or not guard.get("path"):
                raise ValueError("workdir state has no guard dataset")
            guard_path = resolve_workdir_path(self.workdir, guard["path"])
            if sha256_file(guard_path) != guard.get("sha256"):
                raise ValueError("guard dataset digest changed; refusing to compare candidates on a different guard")
            replay = state["data"].get("replay")
            if isinstance(replay, dict) and replay.get("path"):
                replay_path = resolve_workdir_path(self.workdir, replay["path"])
                if sha256_file(replay_path) != replay.get("sha256"):
                    raise ValueError("replay dataset digest changed")
            if not self.dry_run:
                save_state(self.state_path, state)
            return state

        guard_path = unique_path(self.data_dir / "guard", suffix=".jsonl")
        guard_info = guard_dataset(self.baseline.public_validation, guard_path,
                                   rows=self.config.guard.rows, seed=self.config.guard.seed)
        guard_info["path"] = relative_to_workdir(guard_path, self.workdir)
        replay_info = None
        if self.config.experiments.replay_rows > 0:
            replay_path = unique_path(self.data_dir / "replay", suffix=".jsonl")
            replay_info = replay_dataset(self.baseline.public_train, replay_path,
                                         rows=self.config.experiments.replay_rows,
                                         seed=self.config.guard.seed)
            replay_info["path"] = relative_to_workdir(replay_path, self.workdir)

        official_copy = self.models_dir / "official-baseline" / "checkpoint.npz"
        self._ensure_checkpoint(official_copy, self.baseline.checkpoint, official_sha)
        official_rel = relative_to_workdir(official_copy, self.workdir)
        model = self.baseline.load_model(official_copy)
        public_data = self.baseline.load_dataset(self.baseline.public_validation, model.meta["paths"])
        guard_data = self.baseline.load_dataset(resolve_workdir_path(self.workdir, guard_info["path"]),
                                                model.meta["paths"])
        public_metrics = self.baseline.evaluate_model(model, public_data)
        guard_metrics = self.baseline.evaluate_model(model, guard_data)
        state = new_state(
            baseline_root=str(self.baseline.root), profile=self.config.profile,
            baseline_checkpoint_sha256=official_sha, champion_checkpoint=official_rel,
            champion_sha256=official_sha, champion_meta=self._model_meta(model),
            public_validation_metrics=public_metrics, guard_metrics=guard_metrics,
            guard_info=guard_info, replay_info=replay_info, config=self.config,
        )
        state["baseline"]["checkpoint"] = official_rel
        state["baseline"]["public_validation_metrics"] = public_metrics
        state["baseline"]["guard_metrics"] = guard_metrics
        add_event(state, "bootstrap", guard=guard_info, replay=replay_info,
                  public_validation=public_metrics, guard_metrics=guard_metrics)
        save_state(self.state_path, state)
        self.journal.append("bootstrap", checkpoint=official_rel, checkpoint_sha256=official_sha,
                            public_validation=public_metrics, guard=guard_metrics,
                            guard_dataset=guard_info, replay_dataset=replay_info)
        self._say(f"[harness] bootstrap: static RMSE={public_metrics['rmse']:.3f} "
                  f"guard RMSE={guard_metrics['rmse']:.3f}")
        self.swanlab.log({
            "bootstrap/static_public_rmse": public_metrics["rmse"],
            "bootstrap/static_public_mae": public_metrics["mae"],
            "bootstrap/static_guard_rmse": guard_metrics["rmse"],
            "bootstrap/static_guard_mae": guard_metrics["mae"],
            "bootstrap/parameter_count": model.meta["parameter_count"],
            "bootstrap/guard_rows": guard_metrics["n"],
            "bootstrap/public_rows": public_metrics["n"],
            "bootstrap/cpu_threads": int(os.environ.get("BASELINE_CPU_THREADS", "1")),
            "bootstrap/gpu_used": 0,
        }, step=0)
        self.swanlab.log_text("bootstrap/config", json.dumps(
            self.run_config, ensure_ascii=False, allow_nan=False, indent=2), step=0)
        return state

    # ------------------------------------------------------------------ helpers
    def _config_dict(self) -> dict[str, Any]:
        from .config import config_to_dict
        return config_to_dict(self.config)

    def _say(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def _save(self) -> None:
        save_state(self.state_path, self.state)

    def _ensure_checkpoint(self, destination: Path, source: Path, expected_sha: str) -> None:
        destination = Path(destination)
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError(f"checkpoint destination is unsafe: {destination}")
            if sha256_file(destination) != expected_sha:
                raise ValueError(f"checkpoint destination has an unexpected digest: {destination}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            actual = sha256_file(temporary)
            if actual != expected_sha:
                raise ValueError("source checkpoint digest changed during copy")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _model_meta(model) -> dict[str, Any]:
        return {
            "learning_rate": float(model.meta["learning_rate"]),
            "batch_size": int(model.meta["batch_size"]),
            "hidden": int(model.meta["hidden"]),
            "parameter_count": int(model.meta["parameter_count"]),
            "paths": list(model.meta["paths"]),
            "target_mean": float(model.meta["target_mean"]),
            "target_scale": float(model.meta["target_scale"]),
            "feature_mean": [float(value) for value in model.mean],
            "feature_scale": [float(value) for value in model.scale],
            "steps": int(model.meta["steps"]),
            "epochs": int(model.meta["epochs"]),
            "seed": int(model.meta["seed"]),
            "profile": model.meta.get("profile"),
        }

    def _static_checkpoint_path(self) -> Path:
        expected = self.state["baseline"]["official_checkpoint_sha256"]
        stored = self.state["baseline"].get("checkpoint")
        if stored:
            path = resolve_workdir_path(self.workdir, stored)
            if path.is_file() and not path.is_symlink() and sha256_file(path) == expected:
                return path
        if sha256_file(self.baseline.checkpoint) != expected:
            raise ValueError("official static checkpoint digest no longer matches the workdir state")
        return self.baseline.checkpoint

    def _static_model(self):
        if self._static_model_cache is None:
            self._static_model_cache = self.baseline.load_model(self._static_checkpoint_path())
        return self._static_model_cache

    def _load_champion_model(self):
        champion = self.state["champion"]
        path = resolve_workdir_path(self.workdir, champion["checkpoint"])
        if sha256_file(path) != champion.get("sha256"):
            raise ValueError("champion checkpoint digest does not match state.json")
        return self.baseline.load_model(path)

    def _guard_data(self, expected_paths: list[str]):
        guard = self.state["data"]["guard"]
        path = resolve_workdir_path(self.workdir, guard["path"])
        if self._guard_cache is None:
            self._guard_cache = self.baseline.load_dataset(path, expected_paths)
        elif list(self._guard_cache["paths"]) != list(expected_paths):
            raise ValueError("guard feature schema does not match the champion")
        return self._guard_cache

    def _prepare_training_data(self, batch_id: str, batch_dir: Path, validation_data: dict[str, Any],
                               guard_data: dict[str, Any], champion_model) -> dict[str, Any]:
        batch_train = batch_dir / "train.jsonl"
        sources: list[tuple[str, Path]] = []
        replay = self.state["data"].get("replay")
        if batch_train.exists():
            if batch_train.is_symlink() or not batch_train.is_file():
                raise ValueError(f"batch training file is unsafe: {batch_train}")
            if isinstance(replay, dict) and replay.get("path"):
                sources.append(("replay", resolve_workdir_path(self.workdir, replay["path"])))
            sources.append(("batch", batch_train))
        elif isinstance(replay, dict) and replay.get("path"):
            sources.append(("replay", resolve_workdir_path(self.workdir, replay["path"])))
        else:
            sources.append(("public_train", self.baseline.public_train))
        destination = unique_path(self.data_dir / f"train-{safe_name(batch_id, limit=60)}", suffix=".jsonl")
        excluded = set(validation_data["ids"]) | set(guard_data["ids"])
        info = merge_training_sources(sources, destination, excluded_ids=excluded)
        info["path"] = relative_to_workdir(destination, self.workdir)
        validated = self.baseline.load_dataset(destination, champion_model.meta["paths"])
        info["dataset_sha256"] = validated["sha256"]
        info["validated_rows"] = len(validated["ids"])
        return info

    def _materialize_champion_checkpoint(self, source: Path, expected_sha: str, generation: int,
                                         extra_files: tuple[str, ...] = ()) -> tuple[str, str]:
        base = self.models_dir / "champions" / f"generation-{generation:04d}"
        directory = unique_path(base)
        directory.mkdir(parents=True, exist_ok=False)
        destination = directory / "checkpoint.npz"
        self._ensure_checkpoint(destination, Path(source), expected_sha)
        for name in extra_files:
            candidate = Path(source).parent / name
            if candidate.is_file() and not candidate.is_symlink():
                shutil.copy2(candidate, directory / name)
        return relative_to_workdir(destination, self.workdir), sha256_file(destination)

    def _replace_champion_with_static(self, reason: str) -> None:
        old = self.state["champion"]
        static_path = self._static_checkpoint_path()
        expected = self.state["baseline"]["official_checkpoint_sha256"]
        generation = int(old["generation"]) + 1
        rel, digest = self._materialize_champion_checkpoint(static_path, expected, generation)
        metrics = {
            "batch": self.state["baseline"].get("public_validation_metrics"),
            "guard": self.state["baseline"].get("guard_metrics"),
        }
        if metrics["batch"] is None or metrics["guard"] is None:
            static_model = self._static_model()
            public_data = self.baseline.load_dataset(self.baseline.public_validation,
                                                     static_model.meta["paths"])
            guard_info = self.state["data"]["guard"]
            guard_data = self.baseline.load_dataset(
                resolve_workdir_path(self.workdir, guard_info["path"]), static_model.meta["paths"])
            metrics = {"batch": self.baseline.evaluate_model(static_model, public_data),
                       "guard": self.baseline.evaluate_model(static_model, guard_data)}
            self.state["baseline"]["public_validation_metrics"] = metrics["batch"]
            self.state["baseline"]["guard_metrics"] = metrics["guard"]
        self.state["champion"] = {
            "stage_id": "official-baseline", "checkpoint": rel, "sha256": digest,
            "generation": generation, "is_static": True, "promoted_at_batch": None,
            "metrics": metrics, "reference": {**metrics, "source": "repair_to_static"},
            "last_metrics": None, "model": self._model_meta(self._static_model()),
        }
        self.state["previous_champion"] = None
        self.state["counters"]["errors"] += 1
        add_event(self.state, "champion_repair_to_static", reason=reason, generation=generation)
        self.journal.append("champion_repair_to_static", reason=reason, generation=generation)

    # ------------------------------------------------------------------ public API
    def status(self) -> dict[str, Any]:
        return {
            "format": "harness.status.v1",
            "workdir": str(self.workdir),
            "baseline_root": str(self.baseline.root),
            "champion": champion_summary(self.state["champion"]),
            "previous_champion": champion_summary(self.state["previous_champion"])
            if self.state.get("previous_champion") else None,
            "counters": self.state["counters"],
            "batches": self.state["batches"],
            "guard": self.state["data"].get("guard"),
            "replay": self.state["data"].get("replay"),
            "swanlab_enabled": self.swanlab.enabled,
            "swanlab": self.swanlab.describe(),
        }

    def run(self, batches: list[Path], *, max_batches: int | None = None) -> list[dict[str, Any]]:
        selected = batches[:max_batches] if max_batches is not None else batches
        if not selected:
            raise ValueError("no batches to process")
        results = []
        next_index = len(self.state["batches"]) + 1
        try:
            for batch_dir in selected:
                batch_dir = Path(batch_dir).resolve()
                batch_id = batch_dir.name
                if not self.dry_run and batch_id in processed_batches(self.state) and not self.force:
                    previous = next((item for item in self.state["batches"]
                                     if item.get("batch_id") == batch_id), {})
                    validation = batch_dir / "validation.jsonl"
                    recorded = previous.get("validation_sha256")
                    current = (sha256_file(validation)
                               if validation.is_file() and not validation.is_symlink() else None)
                    if recorded and current and recorded != current:
                        raise ValueError(
                            f"batch {batch_id!r} was already processed with different validation data; "
                            "use a new workdir or --force to reprocess")
                    self.journal.append("batch_skipped", batch_id=batch_id, reason="already processed")
                    self._say(f"[harness] skip {batch_id}: already processed")
                    continue
                stored = next((item for item in self.state["batches"]
                               if item.get("batch_id") == batch_id), None)
                if self.force and stored and isinstance(stored.get("batch_index"), int):
                    index = int(stored["batch_index"])
                else:
                    index = next_index
                    next_index += 1
                results.append(self.run_batch(batch_dir, batch_index=index))
            self._log_final()
        finally:
            self.swanlab.finish()
        return results

    def run_batch(self, batch_dir: Path, *, batch_index: int | None = None) -> dict[str, Any]:
        batch_dir = Path(batch_dir)
        batch_id = batch_dir.name
        if batch_index is None:
            batch_index = self._next_batch_index(batch_id)
        started = time.monotonic()
        champion_before = champion_summary(self.state["champion"])
        self.journal.append("batch_started", batch_id=batch_id, batch_index=batch_index,
                            path=str(batch_dir))
        context: dict[str, Any] = {
            "batch_id": batch_id, "batch_index": batch_index, "batch_dir": str(batch_dir),
            "started": started, "champion_before": champion_before, "metadata": None,
            "champion_metrics": None, "static_metrics": None, "drift": None,
            "hypotheses": [], "planned": [], "results": [], "training_data": None,
            "budget_elapsed": 0.0,
        }
        try:
            validation_path = batch_dir / "validation.jsonl"
            if validation_path.is_symlink() or not validation_path.is_file():
                raise ValueError(f"batch validation file is missing or unsafe: {validation_path}")
            metadata = load_optional_batch_metadata(batch_dir)
            context["metadata"] = metadata
            journal_metadata = {"metadata": metadata} if metadata else {}
            self.journal.append("batch_metadata", batch_id=batch_id, **journal_metadata)

            try:
                champion_model = self._load_champion_model()
            except Exception as exc:  # noqa: BLE001 - repair, then keep serving a decision
                self.journal.append("champion_load_failed", batch_id=batch_id,
                                    error=f"{type(exc).__name__}: {exc}")
                if self.dry_run:
                    raise ValueError("champion checkpoint is unusable; dry-run performs no repair") from exc
                self._replace_champion_with_static(f"champion checkpoint unusable: {exc}")
                self._save()
                champion_model = self._load_champion_model()
            champion_meta = champion_model.meta
            validation_data = self.baseline.load_dataset(validation_path, champion_meta["paths"])
            guard_data = self._guard_data(champion_meta["paths"])
            static_model = self._static_model()
            if list(static_model.meta["paths"]) != list(champion_meta["paths"]):
                raise ValueError("static baseline feature schema differs from the champion")

            champion_metrics = {
                "batch": self.baseline.evaluate_model(champion_model, validation_data),
                "guard": self.baseline.evaluate_model(champion_model, guard_data),
            }
            static_metrics = {
                "batch": self.baseline.evaluate_model(static_model, validation_data),
                "guard": self.baseline.evaluate_model(static_model, guard_data),
            }
            context["champion_metrics"] = champion_metrics
            context["static_metrics"] = static_metrics
            reference = self.state["champion"].get("reference") or {}
            reference_batch = reference.get("batch") or self.state["champion"]["metrics"]["batch"]
            drift = analyze_drift(
                x=validation_data["x"], y=validation_data["y"],
                feature_mean=champion_model.mean,
                feature_scale=champion_model.scale,
                feature_paths=list(champion_meta["paths"]),
                target_mean=float(champion_meta["target_mean"]),
                target_scale=float(champion_meta["target_scale"]),
                champion_metrics=champion_metrics["batch"],
                reference_metrics=reference_batch,
                static_metrics=static_metrics["batch"],
                config=self.config.drift,
            )
            context["drift"] = drift.to_dict()
            hypotheses = diagnose(drift, attempts=int(self.state["counters"]["attempts"]),
                                  batch_index=batch_index, config=self.config.experiments)
            context["hypotheses"] = [item.to_dict() for item in hypotheses]
            specs = plan_experiments(drift, hypotheses, champion_meta=champion_meta,
                                     attempts=int(self.state["counters"]["attempts"]),
                                     batch_index=batch_index, config=self.config.experiments)
            context["planned"] = [spec.to_dict() for spec in specs]
            self.journal.append("drift", batch_id=batch_id, **drift.to_dict())
            self.journal.append("diagnosis", batch_id=batch_id,
                                hypotheses=context["hypotheses"], planned=context["planned"])

            if self.dry_run:
                record = self._build_decision_record(
                    context, decision="hold", selected=champion_summary(self.state["champion"]),
                    reason="dry-run：已完成漂移分析、诊断与实验规划，但未执行训练，champion 保持不变。",
                    dry_run=True,
                )
                record["decision_file"] = self._write_decision(record, batch_id, dry_run=True)
                self.journal.append("dry_run", batch_id=batch_id, decision_file=record["decision_file"])
                return record

            training_data = None
            if specs:
                training_data = self._prepare_training_data(batch_id, batch_dir, validation_data,
                                                            guard_data, champion_model)
                context["training_data"] = training_data
                self.journal.append("training_data", batch_id=batch_id, **training_data)
            results = self._run_experiments(context, specs, training_data, validation_data,
                                            guard_data, champion_metrics, static_metrics)
            context["results"] = results

            best = best_passing_candidate(results)
            previous_metrics = None
            if best is not None:
                decision = "promote"
                reason = (f"候选 {best['stage_id']} 通过门控：批次 RMSE 改善 "
                          f"{best['gate'].improvement * 100:.2f}%，guard 未退化且不低于静态基线。")
                selected = self._promote(best, batch_id)
            else:
                rollback, previous_metrics = self._evaluate_rollback(validation_data, guard_data,
                                                                     champion_metrics, static_metrics)
                if rollback.should_rollback:
                    decision = "rollback"
                    reason = rollback.reason
                    selected = self._apply_rollback(rollback, batch_id, static_metrics, previous_metrics)
                else:
                    decision = "hold"
                    reason = self._hold_reason(results, hypotheses, rollback)
                    self.state["counters"]["holds"] += 1
                    self.state["champion"]["last_metrics"] = {
                        "batch_id": batch_id, "batch": champion_metrics["batch"],
                        "guard": champion_metrics["guard"],
                    }
                    add_event(self.state, "hold", batch_id=batch_id, reason=reason)
                    selected = champion_summary(self.state["champion"])

            record = self._build_decision_record(context, decision=decision, selected=selected,
                                                 reason=reason, dry_run=False,
                                                 rollback=rollback.to_dict() if best is None else None)
            record["decision_file"] = self._write_decision(record, batch_id, dry_run=False)
            self._record_batch(record, context)
            context["recorded"] = True
            context["decision_record"] = record
            self._save()
            try:
                self._log_swanlab(record)
            except Exception as exc:  # noqa: BLE001 - logging cannot invalidate a recorded decision
                message = f"{type(exc).__name__}: {exc}"
                try:
                    self.journal.append("swanlab_error", batch_id=batch_id, error=message)
                except Exception:  # noqa: BLE001 - journal failure is handled by the outer guard
                    pass
                self._say(f"[harness] {batch_id}: SwanLab logging failed after the decision: {message}")
            self.journal.append("decision", batch_id=batch_id, decision=decision,
                                reason=reason, selected_model=selected,
                                decision_file=record["decision_file"],
                                counters=self.state["counters"])
            self.journal.append("batch_finished", batch_id=batch_id, decision=decision,
                                elapsed_seconds=round(time.monotonic() - started, 6))
            self._say(self._decision_line(record))
            return record
        except Exception as exc:  # noqa: BLE001 - one bad batch must produce a decision, not stop the stream
            elapsed = time.monotonic() - started
            error = f"{type(exc).__name__}: {exc}"
            if context.get("recorded") and context.get("decision_record"):
                try:
                    self._save()
                except Exception:  # noqa: BLE001 - the decision file already exists
                    pass
                try:
                    self.journal.append("batch_post_decision_error", batch_id=batch_id, error=error,
                                        traceback=traceback.format_exc()[-8000:])
                except Exception:  # noqa: BLE001 - nothing left to persist
                    pass
                self._say(f"[harness] {batch_id}: decision kept; post-decision error: {error}")
                return context["decision_record"]
            self.journal.append("batch_error", batch_id=batch_id, error=error,
                                traceback=traceback.format_exc()[-8000:])
            if self.verbose:
                traceback.print_exc()
            selected = champion_summary(self.state["champion"])
            record = self._build_decision_record(
                context, decision="hold", selected=selected,
                reason=f"批次处理失败，保持当前 champion：{error}",
                dry_run=self.dry_run, error=error,
            )
            record["decision_file"] = self._write_decision(record, batch_id, dry_run=self.dry_run)
            if not self.dry_run:
                self.state["counters"]["errors"] += 1
                self.state["counters"]["holds"] += 1
                self._record_batch(record, context)
                add_event(self.state, "batch_error", batch_id=batch_id, error=error)
                self._save()
            self.journal.append("decision", batch_id=batch_id, decision="hold", reason=record["reason"],
                                selected_model=selected, decision_file=record["decision_file"],
                                counters=self.state["counters"], error=error)
            self.journal.append("batch_finished", batch_id=batch_id, decision="hold",
                                elapsed_seconds=round(elapsed, 6), error=error)
            self._say(self._decision_line(record))
            return record

    # ------------------------------------------------------------------ experiments
    def _run_experiments(self, context, specs, training_data, validation_data, guard_data,
                         champion_metrics, static_metrics) -> list[dict[str, Any]]:
        if not specs:
            return []
        if training_data is None:
            raise ValueError("experiments require prepared training data")
        budget = float(self.config.experiments.time_budget_seconds)
        experiment_timeout = float(self.config.experiments.experiment_timeout_seconds)
        started = time.monotonic()
        champion_checkpoint = resolve_workdir_path(self.workdir, self.state["champion"]["checkpoint"])
        results: list[dict[str, Any]] = []
        for spec in specs:
            elapsed = time.monotonic() - started
            context["budget_elapsed"] = elapsed
            remaining = budget - elapsed
            if remaining <= 0:
                self.journal.append("budget_exhausted", batch_id=context["batch_id"],
                                    elapsed_seconds=round(elapsed, 6), planned=len(specs),
                                    executed=len(results))
                break
            timeout = min(experiment_timeout, max(0.05, remaining))
            stage_id, output_dir = self._allocate_candidate(context["batch_id"], spec)
            self.state["counters"]["attempts"] += 1
            self.journal.append("experiment_started", batch_id=context["batch_id"],
                                stage_id=stage_id, spec=spec.to_dict(), timeout_seconds=timeout)
            self._say(f"[harness] {context['batch_id']} experiment {spec.index}/{len(specs)} "
                      f"lr={spec.learning_rate:g} epochs={spec.epochs} -> {stage_id}")
            try:
                result = self.baseline.train_candidate(
                    train_path=resolve_workdir_path(self.workdir, training_data["path"]),
                    validation_path=Path(context["batch_dir"]) / "validation.jsonl",
                    validation_sha256=validation_data["sha256"],
                    resume_checkpoint=champion_checkpoint,
                    output_dir=output_dir, stage_id=stage_id, epochs=spec.epochs,
                    learning_rate=spec.learning_rate, batch_size=spec.batch_size,
                    threads=int(os.environ.get("BASELINE_CPU_THREADS", "1")),
                    timeout_seconds=timeout,
                )
            except Exception as exc:  # noqa: BLE001 - one candidate failure must not abort the batch
                error = f"{type(exc).__name__}: {exc}"
                self.state["counters"]["failed_experiments"] += 1
                entry = {
                    "spec": spec.to_dict(), "stage_id": stage_id,
                    "training": {"ok": False, "error": error, "duration_seconds": 0.0},
                    "metrics": None, "gate": None, "model_meta": None,
                    "report_rmse": None, "report_mismatch": None, "error": error,
                }
                results.append(entry)
                context["results"] = results
                self._save()
                self.journal.append("experiment_finished", batch_id=context["batch_id"],
                                    stage_id=stage_id, ok=False, gate=None, metrics=None,
                                    error=error, duration_seconds=0.0)
                continue
            entry: dict[str, Any] = {
                "spec": spec.to_dict(), "stage_id": stage_id,
                "training": self._training_summary(result),
                "metrics": None, "gate": None, "model_meta": None,
                "report_rmse": None, "report_mismatch": None, "history": [],
            }
            if result.ok:
                try:
                    candidate_checkpoint = Path(result.checkpoint)
                    candidate_model = self.baseline.load_model(candidate_checkpoint)
                    metrics = {
                        "batch": self.baseline.evaluate_model(candidate_model, validation_data),
                        "guard": self.baseline.evaluate_model(candidate_model, guard_data),
                    }
                    gate = gate_candidate(candidate=metrics, champion=champion_metrics,
                                          static=static_metrics, config=self.config.gate)
                    entry["metrics"] = metrics
                    entry["gate"] = gate
                    entry["model_meta"] = self._model_meta(candidate_model)
                    entry["checkpoint"] = relative_to_workdir(candidate_checkpoint, self.workdir)
                    entry["checkpoint_sha256"] = result.checkpoint_sha256
                    report_rmse = float(result.report["validation_after"]["rmse"])
                    entry["report_rmse"] = report_rmse
                    entry["report_mismatch"] = abs(report_rmse - metrics["batch"]["rmse"])
                    self.journal.append("gate", batch_id=context["batch_id"], stage_id=stage_id,
                                        gate=gate.to_dict(), metrics=metrics)
                    try:
                        entry["history"] = read_history(result.output_dir)
                    except Exception as exc:  # noqa: BLE001 - curves are evidence, not a gate
                        entry["history"] = []
                        entry["history_error"] = f"{type(exc).__name__}: {exc}"
                        self.journal.append("history_error", batch_id=context["batch_id"],
                                            stage_id=stage_id, error=entry["history_error"])
                except Exception as exc:  # noqa: BLE001 - a bad candidate is a failed experiment
                    self.state["counters"]["failed_experiments"] += 1
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    entry["metrics"] = None
                    entry["gate"] = None
                    entry["model_meta"] = None
                    entry.pop("checkpoint", None)
                    entry.pop("checkpoint_sha256", None)
                else:
                    self.state["counters"]["successful_experiments"] += 1
            else:
                self.state["counters"]["failed_experiments"] += 1
                entry["error"] = result.error
            results.append(entry)
            context["results"] = results
            self._save()
            self.journal.append("experiment_finished", batch_id=context["batch_id"],
                                stage_id=stage_id, ok=bool(entry.get("metrics")),
                                gate=entry["gate"].to_dict() if entry.get("gate") else None,
                                metrics=entry.get("metrics"), error=entry.get("error"),
                                duration_seconds=result.duration_seconds)
        context["budget_elapsed"] = time.monotonic() - started
        return results

    @staticmethod
    def _training_summary(result) -> dict[str, Any]:
        report = result.report or {}
        return {
            "ok": result.ok, "error": result.error, "timed_out": result.timed_out,
            "duration_seconds": round(result.duration_seconds, 6),
            "output_dir": result.output_dir,
            "initial_steps": report.get("initial_steps"),
            "final_steps": report.get("final_steps"),
            "initial_epochs": report.get("initial_epochs"),
            "final_epochs": report.get("final_epochs"),
            "training_seconds": report.get("training_seconds"),
            "learning_rate": report.get("learning_rate"),
            "batch_size": report.get("batch_size"),
            "parent_checkpoint_sha256": report.get("parent_checkpoint_sha256"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "stdout_tail": result.stdout_tail if not result.ok else "",
            "stderr_tail": result.stderr_tail if not result.ok else "",
        }

    def _allocate_candidate(self, batch_id: str, spec) -> tuple[str, Path]:
        base = (f"{safe_name(batch_id, limit=40)}-c{spec.index:02d}"
                f"-lr{spec.learning_rate:.6g}-e{spec.epochs}")
        base = safe_name(base, limit=120)
        path = unique_path(self.models_dir / base)
        return path.name, path

    # ------------------------------------------------------------------ decisions
    def _promote(self, best: dict[str, Any], batch_id: str) -> dict[str, Any]:
        old = self.state["champion"]
        generation = int(old["generation"]) + 1
        source = resolve_workdir_path(self.workdir, best["checkpoint"])
        rel, digest = self._materialize_champion_checkpoint(
            source, best["checkpoint_sha256"], generation,
            extra_files=("training.json", "history.jsonl"))
        metrics = best["metrics"]
        champion = {
            "stage_id": best["stage_id"], "checkpoint": rel, "sha256": digest,
            "generation": generation, "is_static": False,
            "promoted_at_batch": batch_id,
            "metrics": {"batch": metrics["batch"], "guard": metrics["guard"]},
            "reference": {"batch": metrics["batch"], "guard": metrics["guard"],
                          "source": f"promotion:{batch_id}"},
            "last_metrics": {"batch_id": batch_id, "batch": metrics["batch"], "guard": metrics["guard"]},
            "model": best["model_meta"],
        }
        self.state["previous_champion"] = old
        self.state["champion"] = champion
        self.state["counters"]["promotions"] += 1
        add_event(self.state, "promote", batch_id=batch_id, stage_id=best["stage_id"],
                  generation=generation, metrics=metrics)
        self._say(f"[harness] promote {best['stage_id']} (generation {generation})")
        return champion_summary(champion)

    def _evaluate_rollback(self, validation_data, guard_data, champion_metrics, static_metrics):
        previous = self.state.get("previous_champion")
        previous_metrics = None
        if previous:
            try:
                path = resolve_workdir_path(self.workdir, previous["checkpoint"])
                if sha256_file(path) == previous.get("sha256"):
                    model = self.baseline.load_model(path)
                    previous_metrics = {
                        "batch": self.baseline.evaluate_model(model, validation_data),
                        "guard": self.baseline.evaluate_model(model, guard_data),
                    }
            except Exception as exc:  # noqa: BLE001 - previous checkpoint is optional evidence
                self.journal.append("previous_champion_unavailable", error=f"{type(exc).__name__}: {exc}")
        decision = rollback_decision(
            champion=self.state["champion"], champion_metrics=champion_metrics,
            static_metrics=static_metrics, previous_metrics=previous_metrics,
            previous_stage_id=previous.get("stage_id") if previous else None,
            config=self.config.gate,
        )
        return decision, previous_metrics

    def _apply_rollback(self, rollback, batch_id: str, static_metrics, previous_metrics):
        old = self.state["champion"]
        generation = int(old["generation"]) + 1
        previous = self.state.get("previous_champion")
        if rollback.target == "previous" and previous and previous_metrics is not None:
            source = resolve_workdir_path(self.workdir, previous["checkpoint"])
            expected = previous["sha256"]
            metrics = previous_metrics
            stage_id = previous["stage_id"]
            model_meta = previous.get("model")
            is_static = bool(previous.get("is_static"))
            if is_static:
                source = self._static_checkpoint_path()
                expected = self.state["baseline"]["official_checkpoint_sha256"]
                metrics = static_metrics
                stage_id = "official-baseline"
                model_meta = self._model_meta(self._static_model())
        else:
            source = self._static_checkpoint_path()
            expected = self.state["baseline"]["official_checkpoint_sha256"]
            metrics = static_metrics
            stage_id = "official-baseline"
            model_meta = self._model_meta(self._static_model())
            is_static = True
        if model_meta is None:
            model_meta = self._model_meta(self.baseline.load_model(source))
        rel, digest = self._materialize_champion_checkpoint(source, expected, generation)
        champion = {
            "stage_id": stage_id, "checkpoint": rel, "sha256": digest,
            "generation": generation, "is_static": is_static,
            "promoted_at_batch": None,
            "metrics": {"batch": metrics["batch"], "guard": metrics["guard"]},
            "reference": {"batch": metrics["batch"], "guard": metrics["guard"],
                          "source": f"rollback:{batch_id}"},
            "last_metrics": {"batch_id": batch_id, "batch": metrics["batch"], "guard": metrics["guard"]},
            "model": model_meta,
        }
        self.state["previous_champion"] = None
        self.state["champion"] = champion
        self.state["counters"]["rollbacks"] += 1
        add_event(self.state, "rollback", batch_id=batch_id, target=rollback.target,
                  stage_id=stage_id, generation=generation, metrics=metrics)
        self._say(f"[harness] rollback to {rollback.target} ({stage_id}, generation {generation})")
        return champion_summary(champion)

    def _hold_reason(self, results, hypotheses, rollback) -> str:
        if not results:
            base = hypotheses[0].statement if hypotheses else "没有可执行的实验，保持 champion。"
            return f"{base}{rollback.reason}"
        executed = [item for item in results if item.get("metrics")]
        if not executed:
            return f"所有实验均执行失败，保持当前 champion。{rollback.reason}"
        best = min(executed, key=lambda item: float(item["metrics"]["batch"]["rmse"]))
        if not best.get("gate"):
            return (f"最佳候选 {best['stage_id']} 批次 RMSE="
                    f"{best['metrics']['batch']['rmse']:.3f}，但独立验证未完成；"
                    f"保持当前 champion。{rollback.reason}")
        failed = best["gate"].failed_checks
        return (f"最佳候选 {best['stage_id']} 批次 RMSE={best['metrics']['batch']['rmse']:.3f}，"
                f"改善 {best['gate'].improvement * 100:.2f}%，未通过门控项：{', '.join(failed)}；"
                f"保持当前 champion。{rollback.reason}")

    def _build_decision_record(self, context, *, decision: str, selected: dict[str, Any],
                               reason: str, dry_run: bool, error: str | None = None,
                               rollback: dict[str, Any] | None = None) -> dict[str, Any]:
        validation_path = Path(context["batch_dir"]) / "validation.jsonl"
        validation_sha256 = None
        if validation_path.is_file() and not validation_path.is_symlink():
            try:
                validation_sha256 = sha256_file(validation_path)
            except ValueError:
                validation_sha256 = None
        batch_info = {
            "id": context["batch_id"], "index": context["batch_index"],
            "path": context["batch_dir"], "validation_sha256": validation_sha256,
            "metadata": context.get("metadata"),
        }
        selected = dict(selected)
        if selected.get("checkpoint"):
            selected["checkpoint_absolute"] = str(
                resolve_workdir_path(self.workdir, selected["checkpoint"]))
        return {
            "format": "harness.decision.v1",
            "workdir": str(self.workdir),
            "batch": batch_info,
            "decided_at": now_utc(),
            "decision": decision,
            "selected_model": selected,
            "champion_before": context["champion_before"],
            "reason": reason,
            "error": error,
            "dry_run": dry_run,
            "drift": context.get("drift"),
            "hypotheses": context.get("hypotheses", []),
            "experiments_planned": context.get("planned", []),
            "experiments": _serialize_experiments(context.get("results", [])),
            "training_data": context.get("training_data"),
            "metrics": {
                "champion_before": context.get("champion_metrics"),
                "static": context.get("static_metrics"),
            },
            "rollback": rollback,
            "budget": {
                "time_budget_seconds": self.config.experiments.time_budget_seconds,
                "experiment_timeout_seconds": self.config.experiments.experiment_timeout_seconds,
                "elapsed_seconds": round(context.get("budget_elapsed") or 0.0, 6),
            },
            "counters": dict(self.state["counters"]),
        }

    def _write_decision(self, record: dict[str, Any], batch_id: str, *, dry_run: bool) -> str:
        prefix = "dry-run-" if dry_run else ""
        base = self.decisions_dir / f"{prefix}{safe_name(batch_id, fallback='batch', limit=60)}"
        path = unique_path(base, suffix=".json")
        write_json_atomic(path, record)
        return relative_to_workdir(path, self.workdir)

    def _record_batch(self, record: dict[str, Any], context) -> None:
        if self.dry_run:
            return
        summary = {
            "batch_id": record["batch"]["id"],
            "batch_index": record["batch"]["index"],
            "validation_sha256": record["batch"].get("validation_sha256"),
            "decided_at": record["decided_at"],
            "decision": record["decision"],
            "drift_severity": (record.get("drift") or {}).get("severity"),
            "drift_summary": (record.get("drift") or {}).get("summary"),
            "champion_before_rmse": (((record.get("metrics") or {}).get("champion_before") or {})
                                     .get("batch") or {}).get("rmse"),
            "static_rmse": (((record.get("metrics") or {}).get("static") or {})
                            .get("batch") or {}).get("rmse"),
            "selected": record["selected_model"],
            "experiments": len(record.get("experiments") or []),
            "attempts_total": self.state["counters"]["attempts"],
            "reason": record["reason"],
            "error": record.get("error"),
            "decision_file": record["decision_file"],
        }
        record_batch(self.state, summary, replace=self.force)

    def _next_batch_index(self, batch_id: str) -> int:
        existing = next((item for item in self.state["batches"]
                         if item.get("batch_id") == batch_id), None)
        if existing and isinstance(existing.get("batch_index"), int):
            return int(existing["batch_index"])
        return len(self.state["batches"]) + 1

    def _decision_line(self, record: dict[str, Any]) -> str:
        metrics = record.get("metrics") or {}
        champion = ((metrics.get("champion_before") or {}).get("batch") or {}).get("rmse")
        static = ((metrics.get("static") or {}).get("batch") or {}).get("rmse")
        drift = (record.get("drift") or {}).get("severity")
        return (f"[harness] {record['batch']['id']}: {record['decision'].upper()} "
                f"champion_rmse={champion if champion is None else round(champion, 3)} "
                f"static_rmse={static if static is None else round(static, 3)} "
                f"drift={drift} attempts={self.state['counters']['attempts']}")

    def _log_swanlab(self, record: dict[str, Any]) -> None:
        if not self.swanlab.enabled:
            return
        step = int(record["batch"]["index"])
        experiments = record.get("experiments") or []
        training_seconds = sum(
            float((item.get("training") or {}).get("training_seconds") or 0.0)
            for item in experiments if isinstance(item, dict))
        champion_model = self.state["champion"].get("model") or {}
        metrics = batch_metrics(record)
        metrics.update({
            "resources/cpu_threads": int(os.environ.get("BASELINE_CPU_THREADS", "1")),
            "resources/process_cpu_count": os.cpu_count() or 1,
            "resources/gpu_used": 0,
            "resources/candidates": len(experiments),
            "resources/training_seconds_total": training_seconds,
            "resources/parameter_count": champion_model.get("parameter_count"),
        })
        self.swanlab.log(metrics, step=step)
        self.swanlab.log(candidate_metrics(record), step=step)
        for epoch, data in sorted(epoch_metrics(record).items()):
            self.swanlab.log(data, step=epoch)
        for key, text in text_records(record):
            self.swanlab.log_text(key, text, step=step)

    def _log_final(self) -> None:
        if not self.swanlab.enabled:
            return
        status = self.status()
        step = max(1, len(status.get("batches") or []) + 1)
        self.swanlab.log(final_metrics(status), step=step)
        summary = {
            "champion": status.get("champion"),
            "counters": status.get("counters"),
            "batches": [{"batch_id": item.get("batch_id"), "decision": item.get("decision"),
                         "drift": item.get("drift_severity"), "decision_file": item.get("decision_file")}
                        for item in status.get("batches") or []],
            "swanlab": self.swanlab.describe(),
        }
        self.swanlab.log_text("final/summary", json.dumps(
            summary, ensure_ascii=False, allow_nan=False, indent=2), step=step)


def _serialize_experiments(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for item in results:
        copy = dict(item)
        gate = copy.get("gate")
        if hasattr(gate, "to_dict"):
            copy["gate"] = gate.to_dict()
        serialized.append(copy)
    return serialized


def discover_batches(stream_root: Path | str) -> list[Path]:
    root = Path(stream_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"stream root must be a regular directory: {root}")
    batches = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name.startswith(".") or child.is_symlink() or not child.is_dir():
            continue
        validation = child / "validation.jsonl"
        if validation.is_file() and not validation.is_symlink():
            batches.append(child.resolve())
    if not batches:
        raise ValueError(f"no batch directories with validation.jsonl under {root}")
    names = [item.name for item in batches]
    if len(set(names)) != len(names):
        raise ValueError("batch directory names must be unique")
    return batches
