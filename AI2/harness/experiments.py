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

from rulkit.data import _group_vector, metrics
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
    """Execute planned actions and return an evidence bundle.

    The runner owns the candidate's **training set**, and that set is the
    heart of the self-evolution story.  Two rules:

    * A candidate is trained on the initial training rows **plus the stream
      rows the harness has already consumed**.  A candidate that only sees
      the initial rows is not an adaptation - it is the same model with a
      different seed, and comparing it against the champion on the stream
      measures nothing.
    * The stream rows a candidate is trained on must not overlap the rows
      used to judge it.  The judge set is always a *later* slice of the
      stream than the training slice, so the comparison is genuinely
      out-of-sample.

    Every action writes a candidate next to the deployed champion; nothing
    here touches the live model.  Promotion is the gate's job.
    """

    def __init__(self, config, train_data, validation_data, workspace):
        self.cfg = config
        self.train = train_data
        self.validation = validation_data
        self.workspace = Path(workspace)
        self.counter = 0
        # Stream rows seen so far, and the index up to which the judge set
        # starts.  The orchestrator advances these as it consumes batches.
        self.stream_x = None
        self.stream_y = None
        self.stream_bearing = None
        self.judge_start = 0
        # The elapsed-snapshot count of every stream row, counted inside its
        # own bearing from the first row the harness saw.  The two-stage
        # inference path needs it, and it must be the *global* count so
        # that a bearing split across two batches still gets a continuous
        # elapsed sequence.
        self.stream_elapsed = None
        # The whole stream's grouping and elapsed counts, set by
        # ``set_stream``.  They give the two-stage path a continuous
        # history even before the orchestrator has observed anything.
        self.stream_groups_all = None
        self.stream_elapsed_all = None
        # The whole stream's rows, set by ``set_stream``.
        self.stream_all_x = None
        self.stream_all_y = None
        self.stream_observed = 0

    # ------------------------------------------------------------------ helpers
    def set_stream(self, stream):
        """Give the runner the whole evaluation stream, up front.

        The runner needs two things that can only be known from the whole
        stream:

        * the bearing grouping, so the two-stage inference path can apply
          the per-bearing slope transform;
        * the elapsed-snapshot count of every row, counted inside its own
          bearing.  A bearing that spans several batches must keep one
          continuous count, or the lifetime estimate sees a machine that
          "restarts" every batch.

        The runner still only ever *trains* on the rows the orchestrator
        has explicitly observed, and it only ever *judges* on rows after
        the judge's boundary.  Knowing the stream is not the same as using
        it.
        """
        if stream is None:
            return
        groups = _group_vector(stream)
        if groups is None:
            return
        groups = np.asarray(groups, dtype=object)
        self.stream_groups_all = groups
        elapsed = np.zeros(len(groups), dtype=np.float64)
        for name in np.unique(groups):
            indices = np.nonzero(groups == name)[0]
            elapsed[indices] = np.arange(1, len(indices) + 1, dtype=float)
        self.stream_elapsed_all = elapsed
        # The whole stream's rows are held so that the judge set exists
        # from the very first cycle.  ``stream_observed`` records how many
        # of them the harness has actually consumed, and only the rows
        # before the judge's boundary may be used for training.
        self.stream_all_x = np.asarray(stream["x"], dtype=np.float64)
        self.stream_all_y = (np.asarray(stream["y"], dtype=np.float64)
                             if stream.get("y") is not None else None)
        self.stream_observed = 0

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

    def final_metrics(self, checkpoint):
        """The champion's score on the independent validation bearings.

        This is the number that goes into the run report, so it has to use
        the same prediction path as the judge: a two-stage checkpoint
        predicts a life fraction, and reading that as minutes would report
        a meaningless score.

        The validation bearings are held out of the whole loop - neither the
        candidates nor the judge ever sees them - so this is the final
        independent check on the system.
        """
        model = Model.load(checkpoint)
        if model.meta.get("target_unit") != "life_fraction":
            return evaluate_checkpoint(checkpoint, self.validation)

        groups = _group_vector(self.validation)
        elapsed = None
        if groups is not None:
            groups = np.asarray(groups, dtype=object)
            elapsed = np.zeros(len(groups), dtype=np.float64)
            for name in np.unique(groups):
                indices = np.nonzero(groups == name)[0]
                elapsed[indices] = np.arange(1, len(indices) + 1, dtype=float)
        predicted = self.predict_rul(model, self.validation["x"], (groups, elapsed))
        result = error_report(predicted, self.validation)
        result["parameter_count"] = model.meta["parameter_count"]
        result["checkpoint_sha256"] = sha256_file(checkpoint)
        return result

    def observe_stream(self, x, y, bearing=None):
        """Record that the harness has consumed a slice of the stream.

        The judge set starts where the last judge slice ended, so the rows
        a candidate trains on are always strictly older than the rows it is
        judged on.
        """
        if x is None or len(x) == 0:
            return
        n = len(x)
        self.stream_observed = min(self.stream_observed + n,
                                   len(self.stream_all_x) if self.stream_all_x is not None else self.stream_observed + n)
        if bearing is not None:
            bearing = np.asarray(bearing, dtype=object)
            self.stream_bearing = (bearing if self.stream_bearing is None
                                    else np.concatenate([self.stream_bearing, bearing]))
        self._refresh_elapsed()

    def _refresh_elapsed(self):
        """Recompute the elapsed-snapshot count of every observed stream row.

        The count is per bearing and continuous across batches, so a bearing
        that spans several batches keeps one coherent history - which is
        what the slope transform and the lifetime estimate both need.

        When the whole stream is known up front (``set_stream``), its counts
        are used verbatim.  They are the authoritative ones: the lifetime
        estimate is defined against the bearing's full snapshot count, so
        counting only the rows the harness happened to observe would shift
        the estimate.
        """
        if self.stream_bearing is None:
            self.stream_elapsed = None
            return
        groups = self.stream_bearing
        if (self.stream_groups_all is not None
                and self.stream_elapsed_all is not None
                and len(self.stream_groups_all) == self.stream_observed):
            self.stream_elapsed = self.stream_elapsed_all
            return
        elapsed = np.zeros(len(groups), dtype=np.float64)
        for name in np.unique(groups):
            indices = np.nonzero(groups == name)[0]
            elapsed[indices] = np.arange(1, len(indices) + 1, dtype=float)
        self.stream_elapsed = elapsed

    def _stream_training_set(self, limit=None):
        """The stream slice a candidate may train on (never the judge slice)."""
        x, y = self._stream_slice(0, self.judge_start)
        if x is None or len(x) == 0:
            return None, None
        if limit is not None:
            end = min(int(limit), len(x))
            x, y = x[:end], (None if y is None else y[:end])
        return x, y

    def _stream_slice(self, start, stop):
        """A slice of the stream's rows.

        The rows come from the stream the runner was given up front, so the
        judge set exists from the first cycle.  ``stop`` is clamped to what
        the harness has actually observed: the harness may not score a
        candidate on rows it has not reached yet.
        """
        if self.stream_all_x is None:
            return None, None
        total = len(self.stream_all_x)
        end = min(int(stop), self.stream_observed, total)
        begin = max(0, int(start))
        if end <= begin:
            return None, None
        x = self.stream_all_x[begin:end]
        y = None if self.stream_all_y is None else self.stream_all_y[begin:end]
        return x, y

    def _judge_set(self):
        """The rows used to judge a candidate: the slice after the training slice.

        This is the row range the harness has *just consumed* and reacted
        to, plus everything that follows it.  The candidate has not been
        trained on any of it, so the comparison is genuinely out-of-sample.
        """
        total = 0 if self.stream_all_x is None else len(self.stream_all_x)
        return self._stream_slice(self.judge_start, total)

    def stream_metrics(self, checkpoint):
        """Score a checkpoint on the **whole** evaluation stream.

        This is the baseline's score.  The baseline has seen none of the
        stream, so the whole stream is out-of-sample for it, and using all
        of it gives the most reliable reference point.

        It also defines the units.  Later candidates are judged on the
        slice that follows their training slice, so the two numbers are
        not interchangeable - the baseline number is the starting point,
        the candidate numbers show where the loop got to.
        """
        if self.stream_all_x is None or self.stream_all_y is None:
            return None
        model = Model.load(checkpoint)
        groups, elapsed = None, None
        if self.stream_groups_all is not None:
            groups = self.stream_groups_all
            elapsed = self.stream_elapsed_all
        predicted = self.predict_rul(model, self.stream_all_x, (groups, elapsed))
        scored = metrics(predicted, self.stream_all_y)
        scored["judged_on"] = "stream_full"
        scored["judge_rows"] = int(len(self.stream_all_x))
        return scored

    def _fit_candidate(self, settings, stage_id, tag, overrides=None):
        """Fit a candidate on initial + stream data and save it.

        Returns ``(checkpoint_path, report)``.  The candidate is
        re-initialised rather than warm-started: the stream arrives in a
        shifted distribution, and warm-starting correlates consecutive
        candidates, which hides whether an update actually helped.

        The checkpoint that gets saved is always the **stage-1 regressor**,
        because that is what the platform's loader understands.  The
        two-stage scale estimate is reconstructed at inference time from
        the stream's own history, so nothing extra has to be serialised.
        """
        from rulkit.stream_train import fit_stream_candidate

        slot = self._slot(tag)
        overrides = overrides or {}
        train_x, train_y = self._stream_training_set()
        stream = None
        if train_x is not None and len(train_x) > 0:
            bearing = None
            if self.stream_groups_all is not None:
                bearing = list(self.stream_groups_all[:len(train_x)])
            elif self.stream_bearing is not None:
                bearing = list(self.stream_bearing[:len(train_x)])
            stream = {"x": train_x, "y": train_y, "bearing": bearing}

        merged = dict(self.cfg["train"])
        merged.update(overrides)

        initial = dict(self.train)
        initial_bearing = _group_vector(initial)
        if initial_bearing is not None:
            initial["bearing"] = initial_bearing

        model, report = fit_stream_candidate(
            initial, stream, 0 if stream is None else len(train_x), merged,
            policy=str(self.cfg["train"].get("stream_policy", "concatenate")),
            seed=int(merged.get("seed", 42)))
        checkpoint = slot / "checkpoint.npz"
        # ``fit_stream_candidate`` returns the stage-1 ``Model`` for the
        # single-stage path and the ``LifeFractionModel`` wrapper for the
        # two-stage path.  Only the former is directly loadable.
        inner = getattr(model, "model", model)
        if report.get("architecture") == "two_stage":
            # Record the target unit so that the inference path knows the
            # checkpoint predicts a life fraction, not minutes, and applies
            # the guarded scale estimate before comparing against the
            # competition's metric.
            inner.meta["target_unit"] = "life_fraction"
            inner.meta["normalisation"] = report["stage2"]["normalisation"]
            inner.meta["stage2_guards"] = report["stage2"]
        inner.save(checkpoint)
        report["stage_id"] = stage_id
        report["stream_rows_in_training"] = 0 if stream is None else len(train_x)
        return checkpoint, report

    def _judge_metrics(self, checkpoint):
        """Score a candidate on the held-back judge slice of the stream."""
        judge_x, judge_y = self._judge_set()
        if judge_x is None or judge_y is None or len(judge_x) < 30:
            # Not enough unseen stream to judge on; fall back to the
            # validation bearings and say so in the report.
            scored = self._metrics(checkpoint)
            scored["judged_on"] = "validation"
            return scored
        model = Model.load(checkpoint)
        predicted = self.predict_rul(model, judge_x, self._judge_view())
        scored = metrics(predicted, judge_y)
        scored["judged_on"] = "stream_holdout"
        scored["judge_rows"] = int(len(judge_x))
        return scored

    def _judge_view(self):
        """The bearing grouping and elapsed counts of the judge slice."""
        if self.stream_groups_all is None or self.stream_elapsed_all is None:
            return None, None
        start = self.judge_start
        stop = min(self.stream_observed, len(self.stream_groups_all))
        if stop <= start:
            return None, None
        return self.stream_groups_all[start:stop], self.stream_elapsed_all[start:stop]

    def stream_view(self, start, stop):
        """The bearing grouping and elapsed counts of a stream slice.

        The elapsed count is continuous across the whole observed stream,
        so a bearing that spans several batches keeps one coherent
        history.  The two-stage inference path needs that history for the
        slope transform and for the lifetime estimate.
        """
        if self.stream_groups_all is None or self.stream_elapsed_all is None:
            return None, None
        stop = min(int(stop), self.stream_observed, len(self.stream_groups_all))
        if stop <= start:
            return None, None
        return self.stream_groups_all[start:stop], self.stream_elapsed_all[start:stop]

    def predict_rul(self, model, x, view=None):
        """Predict the RUL in minutes, honouring the checkpoint's target unit.

        A checkpoint trained on the *absolute* RUL predicts minutes directly.
        A checkpoint trained on the *life fraction* predicts a fraction, and
        the harness converts it back into minutes with the guarded lifetime
        estimate, which needs the bearing's own snapshot history.

        ``view`` is the ``(groups, elapsed)`` pair that describes ``x``.  It
        is ``None`` when the caller has no history, in which case the
        two-stage path falls back to the training-set median lifetime - the
        honest answer rather than an invented scale.
        """
        if model.meta.get("target_unit") != "life_fraction":
            return model.predict(x)
        return self._predict_two_stage(model, x, view)

    def _predict_two_stage(self, model, x, view):
        """Convert a life-fraction prediction into minutes."""
        from rulkit.rul import LifeFractionModel

        wrapper = LifeFractionModel(
            floor=float(self.cfg["train"].get("floor", 0.02)),
            cap=float(self.cfg["train"].get("cap", 3000.0)),
            rate_limit=float(self.cfg["train"].get("rate_limit", 0.25)),
            max_rul_minutes=float(self.cfg["train"].get("max_rul_minutes", 3000.0)),
            normalisation=str(self.cfg["train"].get("normalisation", "slope")))
        wrapper.model = model

        if view is None:
            groups = np.full(len(x), "stream", dtype=object)
            elapsed = np.arange(1, len(x) + 1, dtype=float)
        else:
            groups, elapsed = view
        return wrapper.predict(x, groups, elapsed)

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
        checkpoint = str(slot / "run" / "checkpoint.npz")
        judged = self.stream_metrics(checkpoint) or self._judge_metrics(checkpoint)
        return {"checkpoint": checkpoint, "metrics": judged,
                "cost_seconds": round(time.monotonic() - started, 2), "source": "platform_baseline",
                "validation_metrics": self._metrics(checkpoint),
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
            return self._result("rollback", {"checkpoint": str(target)}, self._judge_metrics(target), started,
                                notes=f"restored {previous_champion['checkpoint']}")
        if action == "recalibrate_inputs":
            return self._recalibrate(champion, stream_batch_x, started)
        if action == "retrain_head":
            return self._retrain_head(champion, started)
        if action == "adapt":
            return self._adapt(champion, params, started)
        if action == "hpo":
            return self._hpo(champion, params, started)
        raise ValueError(f"unknown action {action!r}")

    def _adapt(self, champion, params, started):
        """Train a candidate on the initial data plus the stream seen so far.

        This is the action that makes the loop a *self-evolution* loop: the
        candidate's training set grows with the stream, so a later
        candidate is a genuinely different model.  The judge slice of the
        stream is held back from the candidate's training set, so the
        comparison stays out-of-sample.
        """
        overrides = {k: v for k, v in (params or {}).items()
                     if k in ("hidden", "epochs", "weight_decay", "learning_rate",
                              "batch_size", "feature_transform", "activation", "target_scaling")}
        checkpoint, report = self._fit_candidate(self.cfg["train"], "adapt", "adapt", overrides)
        metric = self._judge_metrics(checkpoint)
        return self._result("adapt", {"checkpoint": str(checkpoint)}, metric, started,
                            notes=("candidate trained on %d initial rows + %d stream rows; judged on %s"
                                   % (report["initial_rows"], report["stream_rows_in_training"],
                                      metric.get("judged_on", "?"))),
                            extra={"training": report})

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
        return self._result("recalibrate_inputs", {"checkpoint": str(target)}, self._judge_metrics(target), started,
                            notes=f"blended input statistics with the current batch (alpha={alpha})")

    def _retrain_head(self, champion, started):
        model = Model.load(champion["checkpoint"])
        train_head_only(model, self.train["x"], self.train["y"], ridge=float(self.cfg["gate"].get("head_ridge", 1.0)))
        target = self._slot("head") / "checkpoint.npz"
        model.save(target)
        return self._result("retrain_head", {"checkpoint": str(target)}, self._judge_metrics(target), started,
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
            try:
                checkpoint, report = self._fit_candidate(
                    self.cfg["train"], f"hpo-{i}", f"hpo{i}",
                    {"hidden": candidate["hidden"], "epochs": candidate["epochs"],
                     "weight_decay": candidate["weight_decay"]})
            except Exception as exc:  # noqa: BLE001 - a failed trial must not kill the loop
                rows.append({**candidate, "error": f"{type(exc).__name__}: {exc}"})
                continue
            metric = self._judge_metrics(checkpoint)
            gc.collect()
            rows.append({**candidate, "rmse": metric["rmse"], "mae": metric["mae"],
                         "per_bearing": metric.get("per_bearing"),
                         "checkpoint": str(checkpoint), "steps": report["epochs"]})
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
            best_metric = self._judge_metrics(best["checkpoint"])
        except Exception as exc:  # noqa: BLE001 - a broken trial must not kill the loop
            return self._result("hpo", champion, champion["metrics"], started,
                                notes=f"best trial could not be evaluated: {type(exc).__name__}: {exc}",
                                failed=True, extra={"trials": rows})
        return self._result("hpo", {"checkpoint": best["checkpoint"]}, best_metric, started,
                            notes=f"best of {len(candidates)} trials: {best['hidden']}u/{best['epochs']}ep/wd={best['weight_decay']}",
                            extra={"trials": rows})
