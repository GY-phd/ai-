# -*- coding: utf-8 -*-
"""Stage 3: execute the planned action and return an evidence bundle.

Every action writes a *candidate* next to the deployed champion; nothing here
touches the live model.  Promotion is the gate's job (stage 4).  Costs are
measured, not estimated: wall-clock training seconds and the number of
gradient updates actually performed.
"""
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from rulkit.data import metrics
from rulkit.evaluate import report as error_report
from rulkit.features import transform
from rulkit.model import Model
from rulkit.train import train_run

from .util import sha256_file, write_json


def _signature(row):
    return (int(row.get("hidden", 0)), int(row.get("epochs", 0)),
            float(row.get("weight_decay", 0.0)), str(row.get("init", "transfer")))


def _passes_screen(champion_metrics, row, gate_cfg):
    """Same two checks the gate applies, minus the budget check, evaluated per trial."""
    gain = (champion_metrics["rmse"] - row["rmse"]) / max(champion_metrics["rmse"], 1e-9) * 100.0
    if gain < gate_cfg["min_gain_pct"]:
        return False
    for bearing, stats in (row.get("per_bearing") or {}).items():
        base = (champion_metrics.get("per_bearing") or {}).get(bearing)
        if not base:
            continue
        delta_pct = (stats["rmse"] - base["rmse"]) / max(champion_metrics["rmse"], 1e-9) * 100.0
        if delta_pct > gate_cfg["max_local_regression_pct_of_global"]:
            return False
    return True


def evaluate_checkpoint(path, data):
    model = Model.load(path)
    predicted = model.predict(data["x"])
    result = error_report(predicted, data)
    result["parameter_count"] = model.meta["parameter_count"]
    result["checkpoint_sha256"] = sha256_file(path)
    return result


def train_head_only(model, x, y, ridge=1.0):
    """Closed-form ridge on the frozen hidden layer: cheap, deterministic, no Adam state."""
    z = model.normalized(x)
    # The activation must match the one the checkpoint was trained with, and the
    # target must be put in the same space the output layer expects.  Both were
    # hard-coded before the activation/target-scaling options existed, which
    # silently fitted the head in the wrong basis.
    pre = z @ model.params["w1"] + model.params["b1"]
    if model.meta.get("activation", "tanh") == "relu":
        hidden = np.maximum(pre, 0.0)
    else:
        hidden = np.tanh(pre)
    if model.meta.get("target_scaling", True):
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    else:
        target = y - model.meta["target_mean"]
    a = np.c_[hidden, np.ones(len(hidden))]
    reg = np.eye(a.shape[1]) * ridge
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(a.T @ a + reg, a.T @ target)
    model.params["w2"] = coef[:-1].astype(np.float64).reshape(-1, 1)
    model.params["b2"] = coef[-1:].astype(np.float64)
    model.second = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.first = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.meta["steps"] = 0
    model.meta["epochs"] = 0
    model.validate()


class ExperimentRunner:
    def __init__(self, config, train_data, validation_data, workspace):
        self.cfg = config
        self.train = train_data
        self.validation = validation_data
        self.workspace = Path(workspace)
        self.counter = 0

    # ------------------------------------------------------------------ helpers
    def _slot(self, tag):
        self.counter += 1
        path = self.workspace / "experiments" / f"{self.counter:02d}-{tag}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _metrics(self, checkpoint):
        result = evaluate_checkpoint(checkpoint, self.validation)
        return {"rmse": result["rmse"], "mae": result["mae"],
                "per_bearing": {k: {"rmse": v["rmse"], "rows": v["rows"]} for k, v in result["per_bearing"].items()},
                "parameter_count": result["parameter_count"]}

    def bootstrap(self, baseline_checkpoint):
        slot = self._slot("bootstrap")
        started = time.monotonic()
        settings = self.cfg["train"]
        report = train_run(self.cfg["data"]["train"], self.cfg["data"]["validation"], slot / "run",
                           resume=baseline_checkpoint, epochs=settings["epochs"], hidden=settings["hidden"],
                           learning_rate=settings["learning_rate"], batch_size=settings["batch_size"],
                           weight_decay=settings["weight_decay"], feature_transform=settings["feature_transform"],
                           stage_id="bootstrap", transfer=True, transfer_epochs=settings["transfer_epochs"],
                           verbose=False)
        return {"checkpoint": str(slot / "run" / "checkpoint.npz"), "metrics": self._metrics(slot / "run" / "checkpoint.npz"),
                "cost_seconds": round(time.monotonic() - started, 2), "source": "platform_baseline",
                "training": {"final_steps": report["final_steps"], "epochs": report["final_epochs"]}}

    # ------------------------------------------------------------------ actions
    def run(self, action, params, champion, stream_batch_x=None, previous_champion=None):
        started = time.monotonic()
        if action == "do_nothing":
            return self._result("do_nothing", champion, champion["metrics"], started, notes="no model change")
        if action == "rollback":
            if not previous_champion:
                return self._result("rollback", champion, champion["metrics"], started, notes="no previous champion to restore", failed=True)
            target = self._slot("rollback") / "checkpoint.npz"
            shutil.copy2(previous_champion["checkpoint"], target)
            return self._result("rollback", {"checkpoint": str(target)}, self._metrics(target), started,
                                notes=f"restored {previous_champion['checkpoint']}")
        if action == "recalibrate_inputs":
            return self._recalibrate(champion, stream_batch_x, started)
        if action == "retrain_head":
            return self._retrain_head(champion, started)
        if action == "hpo":
            return self._hpo(champion, params, started)
        raise ValueError(f"unknown action {action!r}")

    def _result(self, action, candidate, metric, started, notes="", failed=False, extra=None):
        payload = {"action": action, "checkpoint": candidate.get("checkpoint"), "metrics": metric,
                   "cost_seconds": round(time.monotonic() - started, 2), "notes": notes, "failed": failed}
        if extra:
            payload.update(extra)
        return payload

    # ------------------------------------------------------------- sweep memory
    def _trials_path(self):
        return self.workspace / "experiments" / "hpo_trials.json"

    def _tried(self):
        path = self._trials_path()
        if not path.is_file():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("trials", [])
        except Exception:  # noqa: BLE001
            return []

    def _record_trials(self, rows, best):
        history = self._tried()
        combined = history + rows
        scored = [row for row in combined if "rmse" in row]
        global_best = min(scored, key=lambda row: row["rmse"]) if scored else best
        write_json(self._trials_path(), {"trials": combined, "best": global_best})

    def sweep_exhausted(self):
        """True when every configuration in the grid already has a measured result."""
        search = self.cfg["search"]
        grid = [{"hidden": h, "epochs": e, "weight_decay": w, "init": i}
                for h in search["hidden"] for e in search["epochs"]
                for w in search["weight_decay"] for i in search.get("init", ["transfer"])]
        tried = {_signature(row) for row in self._tried()}
        return len([c for c in grid if _signature(c) not in tried]) == 0

    def _recalibrate(self, champion, stream_x, started):
        if stream_x is None:
            return self._result("recalibrate_inputs", champion, champion["metrics"], started,
                                notes="no stream batch supplied", failed=True)
        model = Model.load(champion["checkpoint"])
        mode = model.meta.get("feature_transform", "raw")
        stream = transform(np.asarray(stream_x, dtype=np.float64), mode)
        alpha = 0.5
        model.mean = (1 - alpha) * model.mean + alpha * stream.mean(axis=0)
        model.scale = np.maximum((1 - alpha) * model.scale + alpha * stream.std(axis=0), 1e-9)
        model.validate()
        target = self._slot("recalibrate") / "checkpoint.npz"
        model.save(target)
        return self._result("recalibrate_inputs", {"checkpoint": str(target)}, self._metrics(target), started,
                            notes=f"blended input statistics with the current batch (alpha={alpha})")

    def _retrain_head(self, champion, started):
        model = Model.load(champion["checkpoint"])
        train_head_only(model, self.train["x"], self.train["y"], ridge=float(self.cfg["gate"].get("head_ridge", 1.0)))
        target = self._slot("head") / "checkpoint.npz"
        model.save(target)
        return self._result("retrain_head", {"checkpoint": str(target)}, self._metrics(target), started,
                            notes="closed-form ridge refit of the output layer on the public training set")

    def _hpo(self, champion, params, started):
        search = self.cfg["search"]
        try:
            champion_hidden = int(Model.load(champion["checkpoint"]).meta["hidden"])
        except Exception:  # noqa: BLE001 - fall back to the narrowest grid entry
            champion_hidden = min(search["hidden"])
        trials_left = int(params.get("max_trials", search["max_trials"]))
        candidates = []
        for hidden in search["hidden"]:
            for epochs in search["epochs"]:
                for wd in search["weight_decay"]:
                    for init in search.get("init", ["transfer"]):
                        # A narrower candidate cannot be reached by widening the champion,
                        # so those combinations are skipped instead of failing every trial.
                        if hidden >= champion_hidden:
                            candidates.append({"hidden": hidden, "epochs": epochs,
                                               "weight_decay": wd, "init": init})
        tried = {_signature(row) for row in self._tried()}
        untried = [c for c in candidates if _signature(c) not in tried]
        if not untried:
            return self._result("hpo", champion, champion["metrics"], started,
                                notes="search space exhausted: every configuration has already been measured",
                                failed=True)
        candidates = untried[:max(1, trials_left)]
        rows = []
        best = None
        for i, candidate in enumerate(candidates):
            slot = self._slot(f"hpo{i}")
            try:
                report = train_run(self.cfg["data"]["train"], self.cfg["data"]["validation"], slot / "run",
                                   resume=champion["checkpoint"], epochs=candidate["epochs"], hidden=candidate["hidden"],
                                   learning_rate=self.cfg["train"]["learning_rate"],
                                   weight_decay=candidate["weight_decay"],
                                   feature_transform=self.cfg["train"]["feature_transform"],
                                   stage_id=f"hpo-{i}", transfer=candidate["init"] == "transfer",
                                   reinit=candidate["init"] == "reinit",
                                   transfer_epochs=self.cfg["train"]["transfer_epochs"], verbose=False)
            except Exception as exc:  # noqa: BLE001 - a failed trial must not kill the loop
                rows.append({**candidate, "error": f"{type(exc).__name__}: {exc}"})
                continue
            metric = self._metrics(slot / "run" / "checkpoint.npz")
            gc.collect()
            rows.append({**candidate, "rmse": metric["rmse"], "mae": metric["mae"],
                         "per_bearing": metric.get("per_bearing"),
                         "checkpoint": str(slot / "run" / "checkpoint.npz"), "steps": report["final_steps"]})
            if best is None or metric["rmse"] < best["rmse"]:
                best = rows[-1]
        if best is None:
            return self._result("hpo", champion, champion["metrics"], started, notes="all trials failed", failed=True)
        self._record_trials(rows, best)
        # A sweep must propose a candidate that survives the safety screen; otherwise the
        # gate would reject the winner and the cycle would waste the budget on a runner-up.
        eligible = [row for row in rows if "rmse" in row and _passes_screen(champion["metrics"], row, self.cfg["gate"])]
        if not eligible:
            return self._result("hpo", champion, champion["metrics"], started,
                                notes="no measured candidate passed the local-regression screen "
                                      f"(best raw RMSE {best['rmse']:.1f})",
                                failed=True, extra={"trials": rows})
        screened = min(eligible, key=lambda row: row["rmse"])
        if screened is not best:
            best = screened
        try:
            best_metric = self._metrics(best["checkpoint"])
        except Exception as exc:  # noqa: BLE001 - a broken trial must not kill the loop
            return self._result("hpo", champion, champion["metrics"], started,
                                notes=f"best trial could not be evaluated: {type(exc).__name__}: {exc}",
                                failed=True, extra={"trials": rows})
        return self._result("hpo", {"checkpoint": best["checkpoint"]}, best_metric, started,
                            notes=f"best of {len(candidates)} trials: {best['hidden']}u/{best['epochs']}ep/wd={best['weight_decay']}",
                            extra={"trials": rows})
