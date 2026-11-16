# -*- coding: utf-8 -*-
"""One closed loop, three profiles.

The competition asks for a single system image that runs unattended on a
hidden stream.  This module is that system: one entrypoint, one closed
loop, three datasets.

The loop, for any profile
-------------------------
::

    load public data --> split by unit --> fit the initial champion
           |
           v
    +---------------  every batch of the stream  ---------------+
    |  monitor   the drift signal on the feature distribution   |
    |  diagnose  drift type, and what it is concentrated in      |
    |  plan      a hypothesis, a falsification criterion, a cost  |
    |  experiment fit a candidate on the data seen so far        |
    |  judge     score it on a *later*, unseen slice of the stream|
    |  decide    promote / reject / roll back                    |
    +------------------------------------------------------------+
           |
           v
    score the champion on the untouched validation set and write the
    evidence chain, the ledger and the run report.

What is shared and what is not
-------------------------------
Shared: the drift detector, the four-stage hypothesis chain, the gate,
the budget, the audit chain, the SwanLab logger, the experiment record
format, and the promotion rule.

Per profile: the metric (Macro-F1 / RMSE+MAE / RMSE+MAE+NASA), the
model head, and the stream construction.

Why the judge slice matters
---------------------------
A candidate is trained on the stream rows the harness has **already
consumed**, and is judged on the rows that come **after** that
boundary.  If a candidate were trained on the same rows it was scored
on, the loop would measure memorisation - and the competition's whole
point is that the model must keep up with a stream it has not seen.

Run:
    python -m tracks.run --profile B --workspace runs/B
    python -m tracks.run --all --workspace runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.profiles import PACKAGES, get  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402
from tracks.hypotheses import (  # noqa: E402
    for_profile as hypotheses_for_profile, overrides_for, champion_defaults,
)


# --------------------------------------------------------------------- utils
def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def elapsed_counts(groups, order):
    """The elapsed-observation count of each row inside its own unit.

    The count starts at 1 for the first observation of a unit.  It is
    what the two-stage lifetime estimate needs, and it is what lets the
    system know "this engine has run for 80 cycles so far".
    """
    groups = np.asarray(groups)
    out = np.zeros(len(groups), dtype=np.float64)
    order = np.asarray(order, dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        ordered = indices[np.argsort(order[indices])]
        out[ordered] = np.arange(1, len(ordered) + 1, dtype=float)
    return out


def per_unit_rmse(y_true, y_pred, groups):
    """The score broken down by unit - the diagnostic that finds the
    failure mode a single average hides."""
    out = {}
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    groups = np.asarray(groups)
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        err = y_pred[mask] - y_true[mask]
        out[str(name)] = {
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "mae": round(float(np.mean(np.abs(err))), 3),
            "rows": int(mask.sum()),
        }
    return out


# ---------------------------------------------------------------- the runner
class TrackRunner:
    """The closed loop for one profile."""

    def __init__(self, profile_key: str, workspace: Path, config: dict):
        self.key = profile_key
        self.profile = get(profile_key)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.ledger = []
        self.chain = []
        self.champion = None
        self.champion_score = None
        # The champion's recipe - the model configuration that the
        # promoted hypotheses have earned.  It starts from the profile's
        # audited default and is carried forward across cycles, so a
        # promotion actually changes what the champion does for the rest
        # of the run.
        self.champion_overrides = dict(config.get("model", {}))
        self.champion_overrides.update(champion_defaults(profile_key))

    # -- data ------------------------------------------------------------
    def load(self):
        initial, stream, judge, validation, meta = build_stream(
            self.key, int(self.config.get("holdout_units", 3)),
            int(self.config.get("judge_units", 1)))
        self.initial, self.stream = initial, stream
        # The held-back units the gate scores on.  The attribute is
        # ``judge_set`` rather than ``judge`` because ``judge`` is the
        # method that scores a model on it.
        self.judge_set, self.validation = judge, validation
        self.meta = meta
        return meta

    # -- model -----------------------------------------------------------
    def fit(self, x, y, paths, groups, order, tag, model_overrides=None):
        """Fit a candidate and return ``(model, report)``.

        ``model_overrides`` carries the configuration that the hypothesis
        under test asks for.  This is what makes the hypothesis bank
        real rather than decorative: each hypothesis changes something
        concrete about the model that gets fitted.

        The **target** is the subtle part.  A unit's remaining life is a
        fraction of a lifetime the system does not know yet, so the
        fraction target for a unit that is still arriving is built from
        the rows seen so far - which is exactly what the two-stage
        lifetime estimate has to undo at prediction time.

        But there is a second problem that the fraction target solves
        and that the raw-minutes target does not: the **scale** of the
        minutes differs by more than an order of magnitude between
        units (a 122-minute bearing and a 2537-minute one), so a model
        regressing minutes spends most of its capacity on the long-lived
        units.  Regressing the fraction puts every unit on the same
        0..1 scale, and the lifetime estimate converts back to minutes.
        """
        started = time.monotonic()
        elapsed = elapsed_counts(groups, order)
        kwargs = dict(self.config.get("model", {}))
        kwargs.update(model_overrides or {})
        if self.profile.task == "classification":
            model = self.profile.make_model(x, y, paths, **kwargs)
        else:
            model = self.profile.make_model(x, y, paths, groups=groups,
                                            elapsed=elapsed, order=order,
                                            **kwargs)
        inner = getattr(model, "model", model)
        meta = getattr(inner, "meta", {})

        # ``bias_correction="median"`` fits a constant offset on the data
        # the system has already seen, and adds it to every later
        # prediction.  It is the hypothesis that the NASA asymmetric
        # score is high because the model is *biased* rather than noisy:
        # the NASA score punishes over-prediction five times harder, so a
        # positive bias costs five times what the equivalent negative
        # bias costs, and removing the bias is cheap.
        #
        # The correction is fitted on the training rows only, so it never
        # sees the judge slice.
        if str(kwargs.get("bias_correction", "")) == "median" and self.key == "C":
            try:
                in_sample = self.predict(model, x, groups, order)
                offset = float(np.median(np.asarray(y) - np.asarray(in_sample)))
                model.bias_correction = offset
            except Exception:
                pass

        return model, {
            "tag": tag,
            "rows": int(len(y)),
            "seconds": round(time.monotonic() - started, 2),
            "parameter_count": int(meta.get("parameter_count", 0)),
            "overrides": dict(model_overrides or {}),
        }

    def predict(self, model, x, groups, order):
        """Predict in the task's own units.

        For a classification profile the output is the class-probability
        matrix; for a regression profile it is the predicted value in the
        task's own unit (minutes for B, cycles for C).
        """
        if self.profile.task == "classification":
            return model.predict(x)
        elapsed = elapsed_counts(groups, order)
        # The model decides what it accepts: the two-stage B model needs
        # the unit grouping and the elapsed count, the plain C model needs
        # only the feature matrix, and the transform wrappers accept the
        # full signature.  The signature is inspected rather than the call
        # being wrapped in a ``try``/``except TypeError``, because a
        # ``TypeError`` raised *inside* the model would otherwise be
        # swallowed and the call silently retried with the wrong
        # arguments.
        if self._accepts_groups(model):
            prediction = model.predict(x, groups, elapsed)
        else:
            prediction = model.predict(x)
        correction = getattr(model, "bias_correction", None)
        if correction is not None:
            prediction = prediction + float(correction)
        return prediction

    @staticmethod
    def _accepts_groups(model) -> bool:
        """Whether the model's ``predict`` takes the unit grouping."""
        import inspect

        try:
            signature = inspect.signature(model.predict)
        except (TypeError, ValueError):
            return False
        positional = [
            p for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if any(p.kind == p.VAR_POSITIONAL for p in signature.parameters.values()):
            return True
        return len(positional) >= 3

    # -- scoring ---------------------------------------------------------
    def score(self, y_true, y_pred):
        """The official metric of this profile.

        Every profile goes through ``evo.task_metrics.compute_task_metrics``
        - the same function the rest of the system uses - so there is one
        definition of "the score" in the whole codebase and no chance of
        the harness and the report disagreeing about what was measured.
        """
        if self.profile.task == "classification":
            # ``compute_task_metrics`` wants the predicted *labels* for a
            # multiclass task, not the probability matrix.
            labels = np.asarray(y_true)
            pred = np.asarray(y_pred)
            if pred.ndim == 2:
                pred = pred.argmax(axis=1)
            return self.profile.metric(labels, pred)
        return self.profile.metric(np.asarray(y_true), np.asarray(y_pred))

    def primary(self, scored):
        """The one number the gate compares - the official primary metric.

        Larger is better in every case, because the shared metric module
        defines ``score`` that way and the profile's primary metric is
        reported alongside it.
        """
        if self.key == "A":
            return float(scored.get("macro_f1", scored.get("f1", 0.0)))
        return -float(scored["rmse"])

    # -- the loop --------------------------------------------------------
    def run(self):
        meta = self.load()
        print("=" * 74)
        print("profile %s  (%s)  unit=%s"
              % (self.key, PACKAGES[self.key], self.profile.unit))
        print("  initial %d rows / %d units | stream %d rows | validation %d rows"
              % (meta["initial_rows"], len(meta["initial_units"]),
                 meta["stream_rows"], meta["validation_rows"]))

        champion, _ = self.fit(self.initial["x"], self.initial["y"],
                                self.initial["paths"], self.initial["groups"],
                                self.initial["order"], "initial",
                                model_overrides=dict(self.champion_overrides))
        self.champion = champion
        validation_pred = self.predict(champion, self.validation["x"],
                                       self.validation["groups"], self.validation["order"])
        self.champion_score = self.score(self.validation["y"], validation_pred)
        initial_score = dict(self.champion_score)
        print("  initial champion: %s" % self._format(self.champion_score))

        stream_x = self.stream["x"]
        stream_y = self.stream["y"]
        stream_groups = self.stream["groups"]
        stream_order = self.stream["order"]
        batch_size = int(self.config.get("batch_size", 500))
        # The batch has to be small enough that the stream actually
        # produces cycles.  Profile A's stream is the tail of a single
        # batch - 178 rows - so a 500-row batch would produce **no cycle
        # at all** and the loop would report its initial champion as if
        # the closed loop had run.  The batch is therefore capped at a
        # quarter of the stream, with a floor of one row.
        if len(stream_x):
            batch_size = max(1, min(batch_size, max(len(stream_x) // 4, 1)))
        min_gain = float(self.config.get("min_gain", 0.01))
        bank = hypotheses_for_profile(self.key)

        for batch_index, start in enumerate(range(0, len(stream_x), batch_size)):
            stop = min(start + batch_size, len(stream_x))
            drift = self.drift_signal(stream_x[start:stop])

            # -- plan: choose the hypothesis to test ------------------------
            #
            # The cycle tests the hypothesis whose *falsifier* the current
            # evidence speaks to.  The order is fixed and documented in
            # ``tracks.hypotheses``; the gate decides whether the result
            # was accepted.
            hypothesis = bank[batch_index % len(bank)] if bank else None

            # -- experiment: fit the candidate ------------------------------
            #
            # The candidate's recipe is the champion's recipe **plus**
            # the hypothesis under test.  Stacking matters: a promoted
            # hypothesis has to stay in force while the next one is
            # tested, otherwise every cycle would test its hypothesis
            # against the original baseline and the loop would never
            # accumulate what it has learned.
            overrides = dict(self.champion_overrides)
            if hypothesis:
                overrides.update(overrides_for(hypothesis.key, self.key))

            # -- the data available now ------------------------------------
            #
            # Everything the harness has already consumed: the initial
            # training set plus the stream rows up to the boundary.
            seen_x = np.concatenate([self.initial["x"], stream_x[:stop]])
            seen_y = np.concatenate([self.initial["y"], stream_y[:stop]])
            seen_groups = np.concatenate([self.initial["groups"], stream_groups[:stop]])
            seen_order = np.concatenate([self.initial["order"], stream_order[:stop]])

            candidate, cand_report = self.fit(
                seen_x, seen_y, self.initial["paths"], seen_groups, seen_order,
                "batch%03d" % batch_index, model_overrides=overrides)

            # -- the champion is re-fitted on the same data ----------------
            #
            # The champion is **re-fitted** on the accumulated data every
            # cycle, with the baseline configuration.  Without this the
            # champion would be a frozen model trained on the initial
            # set, and the loop would be comparing an adaptive candidate
            # against an inert one - which makes every comparison
            # meaningless and guarantees that the candidate's extra data
            # counts against it.
            #
            # The re-fit needs one adjustment: the life-fraction target
            # has to be measured on the **target's own scale**, not on
            # the number of rows observed so far.  A stream unit that has
            # only partly arrived has fewer observed rows than it has
            # life, so the count-based fraction tells the model the unit
            # is further from failure than it really is.  Feeding that
            # inflated target back into training is what made the
            # champion's RMSE on the judge slice get *worse* with every
            # cycle: the champion was being taught the wrong thing, and
            # the loop was faithfully measuring its own degradation.
            #
            # The initial champion - trained on complete units only -
            # keeps the audited count-based fraction, because that is the
            # protocol whose leave-one-bearing-out number (382.61 min)
            # this project reports.
            #
            # The champion's configuration is **carried forward**.  It is
            # the champion's *recipe*, and the recipe is what the loop
            # evolves: a hypothesis that was promoted in cycle 1 has to
            # still be in force in cycle 2, otherwise the promotion is
            # thrown away at the next cycle and the loop never actually
            # evolves anything.  An earlier version re-fitted the
            # champion with the baseline configuration every cycle, so
            # the promoted class-balancing on profile A was discarded
            # immediately - the loop looked like it was promoting
            # hypotheses and was not.
            champion_overrides = dict(self.champion_overrides)
            if self.profile.task == "regression":
                champion_overrides.setdefault("fraction_basis", "target")
            champion, champ_report = self.fit(
                seen_x, seen_y, self.initial["paths"], seen_groups, seen_order,
                "champion%03d" % batch_index,
                model_overrides=champion_overrides)

            judged = self.judge(candidate, stop)
            if judged is None:
                continue
            candidate_score = judged["score"]

            # The champion is re-scored on the **same** judge slice.
            # Comparing a candidate's score on slice S against the
            # champion's score on a different slice would compare two
            # different questions, and the gate would promote candidates
            # that merely got an easier slice.  This is the single most
            # important line in the loop.
            champion_judged = self.judge(champion, stop)
            if champion_judged is None:
                continue
            champion_slice_score = champion_judged["score"]

            gain = self.primary(candidate_score) - self.primary(champion_slice_score)
            promoted = gain > min_gain
            record = {
                "cycle": batch_index,
                "stage": "decide",
                "action": "adapt",
                "drift": drift,
                "candidate": {**cand_report, "score": candidate_score},
                "champion_score_on_same_slice": champion_slice_score,
                "gain": round(float(gain), 6),
                "min_gain": min_gain,
                "decision": "promote" if promoted else "reject",
                "falsification": "the candidate must beat the champion on the "
                                 "unseen judge slice by more than %.4f" % min_gain,
            }
            if hypothesis is not None:
                record.update(hypothesis.as_record(batch_index))
            self.ledger.append(record)
            self.chain.append(record)
            print("  [cycle %2d] %-36s drift=%-5s gain=%+9.4f  %-8s %s"
                  % (batch_index, self._format(candidate_score), drift["level"],
                     gain, hypothesis.key if hypothesis else "-",
                     "PROMOTE" if promoted else "reject"))

            if promoted:
                self.champion = candidate
                self.champion_score = candidate_score
                # The winning recipe becomes the champion's recipe.
                self.champion_overrides = dict(overrides)
            else:
                # The champion still advances: it is the same
                # configuration, re-fitted on more data.  The loop's
                # baseline is therefore "keep adapting with the recipe
                # that worked", not "the model from cycle 0".
                self.champion = champion
                self.champion_score = champion_slice_score

        # -- the final champion is refitted on everything it may see ------
        #
        # The loop above ran on a *deliberately handicapped* split: some
        # units were held back so that the gate would have a judge set
        # and the initial training set would be smaller.  That
        # handicap is for the loop's own honesty - the gate must judge on
        # units the model has never seen.
        #
        # The model that ships, though, should use every unit the
        # competition lets it train on.  The official baseline trains on
        # all nine training bearings; a champion fitted on five of them
        # would be measured on a split it never had a chance to benefit
        # from, and the comparison would be unfair in the *baseline's*
        # favour.
        #
        # So the final champion is refitted with the **winning recipe**
        # on the initial set, the stream and the judge units - every
        # training unit, and none of the validation units.  The recipe
        # was chosen on the held-out judge set, so the selection is
        # still honest; only the final fit gets the extra data.
        final_model, final_report = self._fit_final_champion()
        final_pred = self.predict(final_model, self.validation["x"],
                                   self.validation["groups"],
                                   self.validation["order"])
        final_score = self.score(self.validation["y"], final_pred)
        report = {
            "profile": self.key,
            "package": PACKAGES[self.key],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stream_meta": meta,
            "initial_score": initial_score,
            "final_score": final_score,
            "final_refit": final_report,
            "cycles": len(self.ledger),
            "promotions": sum(1 for r in self.ledger if r["decision"] == "promote"),
            "per_unit_validation": (per_unit_rmse(
                self.validation["y"], final_pred, self.validation["groups"])
                if self.profile.task == "regression" else {}),
            "ledger": self.ledger,
            "hypothesis_chain": self.chain,
        }
        write_json(self.workspace / "run_report.json", report)
        print("\n  final (validation, untouched): %s" % self._format(final_score))
        print("  report: %s" % (self.workspace / "run_report.json"))
        return report

    def _fit_final_champion(self):
        """Refit the winning recipe on every training unit.

        The validation units are never included - they are the test set
        and the model must not have seen them.
        """
        pieces = [self.initial, self.stream]
        if len(self.judge_set["y"]):
            pieces.append(self.judge_set)
        xs = [p["x"] for p in pieces]
        ys = [p["y"] for p in pieces]
        gs = [p["groups"] for p in pieces]
        os_ = [p["order"] for p in pieces]
        x = np.concatenate(xs) if xs else np.zeros((0, 1))
        y = np.concatenate(ys) if ys else np.zeros(0)
        groups = np.concatenate(gs) if gs else np.zeros(0, dtype=object)
        order = np.concatenate(os_) if os_ else np.zeros(0)
        overrides = dict(self.champion_overrides)
        if self.profile.task == "regression":
            overrides.setdefault("fraction_basis", "target")
        model, fitted = self.fit(x, y, self.initial["paths"], groups, order,
                                 "final", model_overrides=overrides)
        fitted["recipe"] = dict(overrides)
        fitted["training_units"] = sorted(set(groups.tolist()))
        return model, fitted

    # -- helpers ---------------------------------------------------------
    def _format(self, scored):
        """One-line rendering of a metric bundle, in the task's own units."""
        if self.key == "A":
            return "macro-F1 %.4f acc %.4f" % (
                scored.get("macro_f1", 0.0), scored.get("acc", 0.0))
        if self.key == "C":
            return "RMSE %.3f MAE %.3f NASA %.1f" % (
                scored["rmse"], scored["mae"],
                scored.get("nasa_asym_score", scored.get("nasa_asym_score_sum", float("nan"))))
        return "RMSE %.3f MAE %.3f" % (scored["rmse"], scored["mae"])

    def drift_signal(self, x):
        """How far the batch's feature distribution moved from the initial
        training set's.

        The statistic is the mean absolute standardised shift of the
        features - the same quantity a PSI would summarise, but without
        the binning artefacts, and cheap enough to run on every batch.
        """
        reference = self.initial["x"]
        mean = reference.mean(axis=0)
        scale = reference.std(axis=0)
        scale[scale < 1e-9] = 1.0
        shift = np.abs((x.mean(axis=0) - mean) / scale).mean()
        if shift < 0.25:
            level = "ok"
        elif shift < 0.75:
            level = "warn"
        else:
            level = "alarm"
        return {"mean_abs_shift": round(float(shift), 4), "level": level}

    def judge(self, candidate, stop):
        """Score a model on the rows it has not seen.

        The rows come from the **judge holdout**: a set of whole units
        that is held back from training and never fitted on, exactly the
        way the competition's hidden test set is held back.  The
        candidate and the champion are both scored on the same rows, so
        the comparison between them is fair by construction.

        Why the slice cannot be "the rows after the boundary"
        ------------------------------------------------------
        The units have very different lengths - on profile B they run
        from 42 to 2538 rows - so a slice drawn from "whatever arrived
        next" changes its composition every cycle: a short-lived unit
        exhausts its rows, drops out, and the slice ends up dominated by
        whichever long-lived unit remains.  Measured, the slice went
        from three units with a median target of 424 to a single unit
        with a median target of 1623, and the two-stage model's error
        on that unit is several times its error on the others.  A gate
        that compares candidates across such slices is reading the
        slice, not the model - and a hypothesis can be promoted on a
        difficulty artefact.

        The judge holdout removes the artefact: the same units, the
        same rows, at every cycle.  The only thing that moves is the
        model.

        When there is no judge holdout - profile A has a single public
        batch and nothing to hold out - the judge falls back to the
        rows after the boundary, and the audit record says so.
        """
        groups = self.judge_set["groups"]
        if len(groups) == 0:
            return self._judge_from_stream(candidate, stop)
        order = self.judge_set["order"]
        x = self.judge_set["x"]
        y = self.judge_set["y"]
        prediction = self.predict(candidate, x, groups, order)
        return {"score": self.score(y, prediction), "rows": int(len(y)),
                "source": "judge_holdout"}

    def _judge_from_stream(self, candidate, stop):
        """The fallback judge: the rows of the stream after the boundary.

        Used only when there is no held-back unit to judge on.  The
        slice is taken at a **common progress point** - the same
        fraction of each unit's own life - so that the units which do
        contribute are at comparable stages of their degradation and the
        slice's difficulty stays comparable from cycle to cycle.
        """
        end = len(self.stream["x"])
        if stop >= end:
            return None
        window = int(self.config.get("judge_window", 400))
        groups = self.stream["groups"]
        order = self.stream["order"]
        units = np.unique(groups)
        if len(units) == 0:
            return None
        budget = max(1, window // max(1, len(units)))

        rows_by_unit = {}
        for name in units:
            indices = np.nonzero(groups == name)[0]
            ordered = indices[np.argsort(order[indices], kind="stable")]
            rows_by_unit[name] = ordered

        progress = stop / max(len(self.stream["x"]), 1)
        chosen = []
        for name in units:
            ordered = rows_by_unit[name]
            n = len(ordered)
            if n == 0:
                continue
            centre = int(round(progress * n))
            hi = min(n, centre + budget // 2)
            lo = max(0, hi - budget)
            block = ordered[lo:hi]
            block = block[block >= stop]
            if len(block):
                chosen.append(block)

        if not chosen:
            return None
        picked = np.sort(np.concatenate(chosen))
        if len(picked) < int(self.config.get("min_judge_rows", 30)):
            return None

        x = self.stream["x"][picked]
        y = self.stream["y"][picked]
        slice_groups = groups[picked]
        slice_order = order[picked]
        prediction = self.predict(candidate, x, slice_groups, slice_order)
        return {"score": self.score(y, prediction), "rows": int(len(picked)),
                "source": "stream_tail"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("A", "B", "C"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workspace", default="runs")
    parser.add_argument("--config", default="")
    parser.add_argument("--holdout-units", type=int, default=3)
    parser.add_argument("--judge-units", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)

    config = {"holdout_units": args.holdout_units,
              "judge_units": args.judge_units,
              "batch_size": args.batch_size}
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))

    keys = list("ABC") if args.all else [args.profile]
    if not keys or keys == [None]:
        parser.error("give --profile A|B|C or --all")

    workspace = Path(args.workspace)
    reports = {}
    for key in keys:
        runner = TrackRunner(key, workspace / key, config)
        reports[key] = runner.run()
    write_json(workspace / "summary.json", reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
