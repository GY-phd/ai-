# -*- coding: utf-8 -*-
"""The round's Controller: the steps of one closed-loop round.

The round as a sequence of steps
---------------------------------
Each step below is a method, and each one does one thing:

=========================  ====================================================
step                       what it does
=========================  ====================================================
``open_parent``            read the parent checkpoint, check its digest
``read_public``            read the public training and validation rows
``plan``                   choose the hypothesis for this round
``measure_parent``         score the parent on the validation split
``build_candidate``        apply the hypothesis and train
``measure_candidate``      score the candidate on the same split
``decide``                 compare the two and say why
``record``                 write the ledger entry and the SwanLab record
``consult_the_model``      ask the host's model to review the reading
``submit``                 hand the candidate over, or keep the parent
``finish``                 declare the round's outcome
=========================  ====================================================

The order is not negotiable.  The parent is measured **before** the
candidate is built, so the comparison is against the parent's own
number; and the decision is made **before** the model is consulted, so
the model's reply is a review of a decision rather than the decision
itself.

What the Controller owns, and what it borrows
----------------------------------------------
The Controller owns the **protocol**: it is the only layer that reads
``rnd``, that talks to the host's model, and that submits artefacts.
It borrows the **data** from the Model layer and the **profile** from
the profile's adapter.

What it deliberately does not own
---------------------------------
The **forward pass**.  The candidate is a reference-format checkpoint
and the measurement runs the reference forward, so the platform's own
scorer sees exactly what this code sees.

Why the steps are methods and not one long function
---------------------------------------------------
Because a round is a *process*, and a process has steps that can fail
independently.  When a round goes wrong the question is always "which
step?", and a stack of named steps answers it.  The steps also make the
round **replayable**: the offline rehearsal runs the same controller
the platform runs, so a difference between the two is a difference in
the environment and not in the logic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

from ...adapters import get as get_adapter
from ...data import read_rows
from ...metrics import (
    primary_metric_name,
    primary_metric_value,
    score_metrics,
)
from ...strategies import build_plan, epoch_budget, prepare
from ...swanlab_log import SubmissionSwanLab

# The **reference** model (``contest_baselines/model.py``) is the platform's
# artefact format and its forward pass.  It is not the Model layer.
from ...model import Model, checkpoint_digest

# The Model layer's tables.
from ..model.checkpoint import Checkpoint
from ..model.ledger import LedgerEntry
from ..model.metrics import (
    BETTER,
    CONFLICT,
    THE_SAME,
    WORSE,
    MetricRecord,
    compare,
    compare_joint,
)
from ..model.state import RoundState

# The host's model service can be temporarily unavailable, and a gateway can
# reject an option it does not know.  Neither is a property of this
# submission, so the model call is retried instead of ending the round.
MODEL_CALL_ATTEMPTS = 3
MODEL_CALL_BACKOFF = 3.0

# The reply is a review of one measurement, so it needs room to finish but
# not the whole authorized budget.
MODEL_TOKEN_BUDGET = 1024


def log(message):
    """Log to stderr.  stdout carries only the protocol's JSONL."""
    print("[contest] %s" % message, file=os.sys.stderr, flush=True)


class RoundController:
    """One closed-loop round, as a sequence of steps."""

    def __init__(self, rnd):
        self.rnd = rnd
        self.started = time.monotonic()
        self.dataset_key = rnd.dataset_key
        self.task = rnd.task.get("task")
        self.adapter = get_adapter(self.dataset_key)
        self.workspace = Path(os.environ.get("CONTEST_WORK", "/workspace"))
        self.parent = None          # the Checkpoint
        self.parent_name = None
        self.parent_path = None
        self.candidate = None       # the Checkpoint
        self.train = None
        self.validation = None
        self.hypothesis = None
        self.changed = {}
        self.epochs = 0
        self.seed = 42
        self.parent_record = None
        self.candidate_record = None
        self.verdict = THE_SAME
        self.adopted = False
        self.reason = ""
        self.state = None
        self.state_dir = None
        self.candidate_dir = None
        self.tracker = None
        self.review = ""
        self.receipt = None
        self.history = []
        self.selected = None

    # ---------------------------------------------------------------- setup
    def open_parent(self):
        """Read the parent checkpoint and check that it matches this task.

        The file is opened through ``rnd.parent.path``, which verifies
        the digest the platform declared.  Nothing is searched for in
        ``/app`` and no hash is fabricated.
        """
        self.parent_name = self._parent_checkpoint_name()
        self._parent_root()
        path = self.rnd.parent.path(self.parent_name)
        model = Model.load(path)
        if self.task and model.meta.get("task") != self.task:
            raise RuntimeError("the parent checkpoint is %s but the task is %s"
                               % (model.meta.get("task"), self.task))
        if model.meta.get("paths") is None:
            raise RuntimeError("the parent checkpoint declares no feature schema")
        self.parent = Checkpoint(model)
        self.parent_path = path
        log("parent checkpoint %s: task=%s hidden=%d epochs=%d steps=%d features=%d"
            % (self.parent_name, self.parent.task, self.parent.hidden,
               self.parent.epochs, model.meta["steps"], len(self.parent.paths)))
        return self.parent

    def _parent_checkpoint_name(self):
        """The parent checkpoint's path *within* the parent artefact.

        The parent declares its own files; the checkpoint is the one that
        looks like a checkpoint.  Nothing is searched for in ``/app`` and
        no hash is fabricated: the file is opened through
        ``rnd.parent.path``, which checks the digest the platform
        declared.
        """
        declared = list(self.rnd.parent.declared_paths())
        candidates = [name for name in declared if name.endswith(".npz")]
        if not candidates:
            candidates = [name for name in declared if "checkpoint" in name.lower()]
        if not candidates:
            raise RuntimeError("the parent artifact declares no checkpoint file")
        # The shortest name is the root-level one (``checkpoint.npz``).
        candidates.sort(key=lambda name: (name.count("/"), len(name)))
        return candidates[0]

    def _parent_root(self):
        """Where the parent artefact is mounted.

        The platform mounts it at ``/previous``.  The offline test
        harness runs on a machine where that path does not exist, so it
        may point the mount at a local copy through
        ``CONTEST_PARENT_ROOT``.  Nothing else changes: the file is
        still opened through ``rnd.parent.path``, which checks the
        digest the platform declared.
        """
        override = os.environ.get("CONTEST_PARENT_ROOT")
        if override:
            self.rnd.parent.root = Path(override)
        return self.rnd.parent.root

    def read_public(self):
        """Read the public training and validation projections."""
        model = self.parent.model
        classes = list(model.meta.get("classes") or [])
        self.train = read_rows(self.rnd.data.train_path, model.meta["task"],
                                classes)
        self.validation = read_rows(self.rnd.data.validation_path,
                                     model.meta["task"], classes,
                                     self.train["paths"])
        if set(self.train["ids"]) & set(self.validation["ids"]):
            raise RuntimeError("the training and validation sample IDs overlap")
        log("public data: %d training rows, %d validation rows, %d features"
            % (len(self.train["ids"]), len(self.validation["ids"]),
               len(self.train["paths"])))
        return self.train, self.validation

    # ----------------------------------------------------------------- plan
    def plan(self):
        """Choose the hypothesis for this round.

        The plan is a function of what has already been tried, which is
        what makes the ten rounds an experiment rather than ten
        repetitions.
        """
        self.history = list(self.rnd.history())
        self.hypothesis = build_plan(self.dataset_key, self.rnd.round_index,
                                     self.history, adapter=self.adapter)
        log("round %d: hypothesis %r" % (self.rnd.round_index, self.hypothesis.key))
        return self.hypothesis

    # ------------------------------------------------------------- measuring
    def _measure(self, checkpoint, data):
        """Score a checkpoint on a labelled split."""
        model = checkpoint.model
        if model.meta["task"] == "classification":
            predicted = model.predict(data["x"]).argmax(axis=1)
        else:
            predicted = model.predict(data["x"])
        values = score_metrics(model.meta["task"], data["y"], predicted,
                               profile=self.dataset_key)
        name = primary_metric_name(model.meta["task"])
        view = self.adapter.selection_view(data, predicted)
        selection_name = selection_value = None
        if view:
            # A protocol-aligned slice: profile C's hidden test scores
            # truncated snapshots, not every cycle of a complete
            # sequence.  The full-split primary metric stays in
            # ``values`` and the decision applies a guard on it.
            values[view["name"]] = view["value"]
            values[view["name"] + "_n"] = view["n"]
            selection_name, selection_value = view["name"], view["value"]
        return MetricRecord(model.meta["task"], values, name,
                            primary_metric_value(model.meta["task"], values),
                            selection_name=selection_name,
                            selection_value=selection_value)

    def measure_parent(self):
        """Score the parent on the public validation split."""
        self.parent_record = self._measure(self.parent, self.validation)
        log("parent %s = %.4f" % (self.parent_record.primary_name,
                                  self.parent_record.primary_value))
        if self.parent_record.selection_name != self.parent_record.primary_name:
            log("parent %s = %.4f" % (self.parent_record.selection_name,
                                      self.parent_record.selection_value))
        return self.parent_record

    def measure_candidate(self):
        """Score the candidate on the same split, under the same conditions."""
        self.candidate_record = self._measure(self.candidate, self.validation)
        log("candidate %s = %.4f" % (self.candidate_record.primary_name,
                                     self.candidate_record.primary_value))
        if self.candidate_record.selection_name != self.candidate_record.primary_name:
            log("candidate %s = %.4f" % (self.candidate_record.selection_name,
                                         self.candidate_record.selection_value))
        return self.candidate_record

    # -------------------------------------------------------------- building
    def build_candidate(self):
        """Apply the hypothesis and train the candidate.

        The candidate starts as an **exact copy** of the parent: the
        weights, the Adam moments, the epoch counter and the
        normalisation state all carry over.  Only the training procedure
        changes.
        """
        model = Model.load(self.parent_path)
        self.seed = int(model.meta.get("seed", 42))
        x, y, self.changed = prepare(self.hypothesis, model, self.train,
                                     self.dataset_key, self.seed,
                                     adapter=self.adapter)
        self.epochs = epoch_budget(self.hypothesis, self.rnd.round_index,
                                   self.rnd.wall_seconds, adapter=self.adapter)
        log("applying %r: %s" % (self.hypothesis.key,
                                 json.dumps(self.changed, ensure_ascii=False)))
        model.fit(x, y, self.epochs)
        self.candidate = Checkpoint(model)
        return self.candidate

    # -------------------------------------------------------------- deciding
    def decide(self):
        """Compare the two measurements and say why.

        The falsifier is here: an improvement that the hypothesis did
        **not** predict is treated as a refutation, not as a
        confirmation.  A change that improved the score for a reason
        other than the one claimed is not evidence for the claim.

        The verdict is a **conjunction** when the profile exposes a
        protocol-aligned selection slice: the full split must not
        regress *and* the slice must improve.  See
        :func:`contest_baselines.layers.model.metrics.compare_joint`
        for why neither metric alone is enough - the platform's own
        2026-11-22 run is the evidence.
        """
        self.verdict = compare_joint(self.parent_record, self.candidate_record,
                                     self.parent.task)
        guard = self._full_split_guard()
        self.guard = guard
        improved = self.verdict == BETTER
        consistent = self._consistent_with_expectation(improved)
        self.adopted = bool(improved and consistent and guard["passed"])

        if improved and not guard["passed"]:
            self.reason = (
                "the protocol-aligned metric improved (%s %s -> %s) but the "
                "full-split %s regressed by %+.1f%% (tolerance +%.1f%%), so "
                "the candidate is not adopted"
                % (self.parent_record.selection_name,
                   _fmt(self.parent_record.selection_value),
                   _fmt(self.candidate_record.selection_value),
                   self.parent_record.primary_name, guard["rel"] * 100.0,
                   guard["limit"] * 100.0))
        elif improved and not consistent:
            self.reason = (
                "the score improved (%s %.4f -> %.4f) but not in the direction "
                "the hypothesis predicted, so the change is not adopted"
                % (self.parent_record.selection_name,
                   self.parent_record.selection_value,
                   self.candidate_record.selection_value))
        elif improved:
            self.reason = (
                "the hypothesis is confirmed: %s moved from %.4f to %.4f"
                % (self.parent_record.selection_name,
                   self.parent_record.selection_value,
                   self.candidate_record.selection_value))
            if self.candidate_record.selection_name != self.candidate_record.primary_name:
                self.reason += (
                    " (full %s %.4f -> %.4f, within the +%.1f%% guard)"
                    % (self.parent_record.primary_name,
                       self.parent_record.primary_value,
                       self.candidate_record.primary_value,
                       guard["limit"] * 100.0))
        elif self.verdict == THE_SAME:
            self.reason = "the change left the score unchanged, so it is not adopted"
        elif self.verdict == CONFLICT:
            self.reason = (
                "the two readings disagree (%s %.4f -> %.4f, full %s %.4f -> "
                "%.4f), so the candidate is not adopted"
                % (self.parent_record.selection_name,
                   self.parent_record.selection_value,
                   self.candidate_record.selection_value,
                   self.parent_record.primary_name,
                   self.parent_record.primary_value,
                   self.candidate_record.primary_value))
        else:
            self.reason = (
                "the hypothesis is refuted: %s moved from %.4f to %.4f"
                % (self.parent_record.selection_name,
                   self.parent_record.selection_value,
                   self.candidate_record.selection_value))
        log("decision: %s" % self.reason)
        return self.adopted, self.reason

    def _consistent_with_expectation(self, improved):
        """Does the measurement look like the hypothesis predicted?

        Every hypothesis in the bank predicts that the primary metric
        moves in one direction.  A change that moves the metric the
        *other* way cannot confirm the hypothesis, no matter how much the
        number moved, so it is reported as a refutation.
        """
        if not improved:
            return False
        before = self.parent_record.selection_value
        after = self.candidate_record.selection_value
        if self.parent.task == "classification":
            return after > before
        return after < before

    def _full_split_guard(self):
        """Bound how far the full-split metric may regress.

        The selection slice is closer to the hidden protocol, but the
        hidden pack is a different dataset.  A candidate that wins the
        slice by abandoning the full split is fragile, so the round keeps
        the parent unless the full-split primary metric stays within the
        adapter's tolerance.  Profiles without a selection view are not
        guarded (the full split *is* their protocol).
        """
        before = self.parent_record
        after = self.candidate_record
        if after.selection_name == after.primary_name:
            return {"active": False, "passed": True, "limit": 0.0, "rel": 0.0}
        limit = float(self.adapter.selection_guard())
        base = float(before.primary_value)
        rel = (float(after.primary_value) - base) / base if base > 0 else 0.0
        return {"active": True, "passed": rel <= limit, "limit": limit,
                "rel": rel, "before": base, "after": float(after.primary_value)}

    # -------------------------------------------------------------- recording
    def record(self):
        """Write the ledger entry and the SwanLab record.

        Two records, for two different readers: the ledger is the
        round's own evidence and travels to the next round as the
        native state; the SwanLab record is the experiment log in
        SwanLab's shape, and it is written **locally** and copied into
        both the candidate directory and the state, so the platform's
        evaluation machine collects it whether or not the candidate was
        adopted.
        """
        self.state_dir = self.rnd.state_workspace(str(self.workspace / "state"))
        self.state = RoundState(self.state_dir)

        entry = LedgerEntry(
            round_index=self.rnd.round_index,
            dataset_key=self.dataset_key,
            task=self.parent.task,
            hypothesis=self.hypothesis.key,
            change=json.dumps(self.changed, ensure_ascii=False),
            before=self.parent_record.values,
            after=self.candidate_record.values,
            adopted=self.adopted,
            reason=self.reason,
            claim=self.hypothesis.claim,
            change_summary=self.hypothesis.change,
            falsifier=self.hypothesis.falsifier,
            expected_effect=self.hypothesis.expect,
            applied=self.changed,
            primary_metric=self.parent_record.primary_name,
            epochs=self.epochs,
            learning_rate=float(self.candidate.model.meta["learning_rate"]),
            parent_checkpoint_sha256=self.rnd.parent.checkpoint_sha256,
            stage_id=self.rnd.stage_id,
        )
        self.state.append_ledger(entry)

        self.tracker = SubmissionSwanLab(
            self.workspace,
            run_name="zhixun-engine-%s-round-%d" % (self.dataset_key,
                                                     self.rnd.round_index),
            config={"dataset_key": self.dataset_key, "task": self.parent.task,
                    "adapter": self.adapter.label, "round": self.rnd.round_index,
                    "hypothesis": self.hypothesis.key, "epochs": self.epochs},
        )
        name = self.parent_record.primary_name
        self.tracker.log(
            {
                "val_%s_before" % name: self.parent_record.primary_value,
                "val_%s_after" % name: self.candidate_record.primary_value,
                "val_%s_delta" % name: (self.candidate_record.primary_value
                                         - self.parent_record.primary_value),
                "score_before": self.parent_record.values.get("score"),
                "score_after": self.candidate_record.values.get("score"),
                "epochs": self.epochs,
                "learning_rate": float(self.candidate.model.meta["learning_rate"]),
                "batch_size": int(self.candidate.model.meta["batch_size"]),
                "hidden": self.candidate.hidden,
                "adopted": bool(self.adopted),
                "train_rows": len(self.train["ids"]),
                "validation_rows": len(self.validation["ids"]),
                "seconds": round(time.monotonic() - self.started, 2),
            },
            context={
                "stage": "round",
                "dataset_key": self.dataset_key,
                "task": self.parent.task,
                "adapter": self.adapter.label,
                "round": self.rnd.round_index,
                "hypothesis": self.hypothesis.key,
                "claim": self.hypothesis.claim,
                "change": self.hypothesis.change,
                "falsifier": self.hypothesis.falsifier,
                "expected_effect": self.hypothesis.expect,
                "applied": self.changed,
                "decision": self.reason,
                "parent_checkpoint_sha256": self.rnd.parent.checkpoint_sha256,
            },
        )
        self.tracker.finish()
        # The record travels with the state that the next round inherits.
        self._copy_record()
        return entry

    def _copy_record(self):
        """Copy the SwanLab record into the candidate and the state.

        The candidate directory may not exist yet when the record is
        written, so the copy into it is attempted again once the
        directory has been created.  A missing copy must never cost a
        measured round, so a failure here is swallowed.
        """
        if self.tracker is None or not self.tracker.path.is_file():
            return
        if self.state_dir is not None:
            try:
                shutil.copyfile(self.tracker.path,
                                self.state_dir / self.tracker.path.name)
            except OSError:
                pass
        if self.candidate_dir is not None and self.candidate_dir.is_dir():
            try:
                shutil.copyfile(self.tracker.path,
                                self.candidate_dir / self.tracker.path.name)
            except OSError:
                pass

    # ------------------------------------------------------------ the model
    def consult_the_model(self):
        """Ask the host's model to review this round's reading.

        The round must contain at least one real model call, and the
        call has to be a genuine part of the decision rather than a
        formality: the model is asked to check the reading of the
        measurement, and its reply is recorded in the round's reason.
        """
        brief = self._model_brief()
        self.receipt, self.review = self._call_model(brief)
        log("model call %s replied with %d characters"
            % (self.receipt.call_id, len(self.review)))
        return self.receipt, self.review

    def _model_brief(self):
        """The message sent to the platform's model.

        Only this round's own state is sent: the hypothesis, the
        measurement, and the decision.  The official prompt is **not**
        repeated here - the host attaches it to the request, and sending
        it again would duplicate it.
        """
        return (
            "本轮实验记录（数据集 %s，第 %d 轮，任务 %s）。\n"
            "假设：%s\n"
            "所做的改动：%s\n"
            "证伪条件：%s\n"
            "公开验证集上的主指标：父模型 %.4f -> 候选 %.4f。\n"
            "当前决策：%s。\n"
            "请只依据以上真实测量，判断这个读数是否支持该假设，并指出下一步最值得验证的一个改动。"
            % (self.dataset_key, self.rnd.round_index, self.parent.task,
               self.hypothesis.claim, self.hypothesis.change,
               self.hypothesis.falsifier,
               self.parent_record.primary_value,
               self.candidate_record.primary_value,
               "采纳候选" if self.adopted else "保留父模型")
        )

    def _call_model(self, brief):
        """Ask the host's model for a review, and return ``(receipt, text)``.

        Three things about the call are handled here rather than left to
        chance.

        **The reasoning trace is turned off.**  The host's models are
        reasoning models: with the trace enabled, the whole token budget
        is spent on the trace, the reply comes back empty, and a large
        model can take minutes to do it - long enough for the gateway to
        report the model service as unavailable.  Turning the trace off
        makes the reply arrive as actual text, quickly.

        **A host that does not know the option is asked without it**,
        which is why there is more than one attempt.

        **A temporary failure is retried.**  The model service is
        shared, so it can be momentarily unavailable.  That is not a
        property of this submission, and ending the whole round on it
        would throw away a completed measurement.

        The call is never skipped: the round's contract is that it
        contains a real model call, so the last attempt is allowed to
        raise.
        """
        messages = [{"role": "user", "content": brief}]
        ceiling = int(self.rnd.max_output_tokens)
        budget = min(MODEL_TOKEN_BUDGET, ceiling)
        attempts = (
            # 1. The trace off: the reply arrives as text, quickly.
            ({"max_tokens": budget, "temperature": 0, "enable_thinking": False}, False),
            # 2. Without the option, in case the host does not know it.
            ({"max_tokens": budget, "temperature": 0}, False),
            # 3. Without the option and with the full authorized budget.
            ({"max_tokens": ceiling, "temperature": 0}, False),
        )
        last_error = None
        for index, (options, _) in enumerate(attempts):
            if index:
                time.sleep(MODEL_CALL_BACKOFF * index)
            try:
                receipt = self.rnd.model(messages, **options)
            except Exception as error:
                last_error = error
                log("model call attempt %d failed: %s: %s"
                    % (index + 1, type(error).__name__, error))
                continue
            text = (receipt.text or "").strip()
            if text:
                return receipt, text
            # An empty reply is the reasoning-trace symptom: the budget
            # went into the trace and nothing came back.  The next attempt
            # asks without the trace.
            last_error = RuntimeError("the model replied without any text")
            log("model call attempt %d returned no text" % (index + 1))
        raise last_error if last_error else RuntimeError("the model call failed")

    # -------------------------------------------------------------- submitting
    def submit(self):
        """Hand the candidate over, or keep the parent.

        A candidate that was not adopted must not be uploaded; its
        directory is discarded so the round leaves no orphan.
        """
        if self.candidate_dir is None:
            self.candidate_dir = self.workspace / ("candidate-round-%d"
                                                    % self.rnd.round_index)
        if self.candidate_dir.exists():
            shutil.rmtree(self.candidate_dir)
        self.candidate_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = self.candidate.save(self.candidate_dir)
        checkpoint_digest(checkpoint_path)
        code_paths = self._copy_code(self.candidate_dir)
        # The SwanLab record was written before the candidate directory
        # existed, so it is copied in now that the directory does.
        self._copy_record()

        if self.adopted:
            submitted = self.rnd.submit_candidate(
                self.candidate_dir,
                hypothesis=self.hypothesis.claim,
                change_summary="%s Applied: %s. Validation %s moved from %.4f to %.4f."
                               % (self.hypothesis.change,
                                  json.dumps(self.changed, ensure_ascii=False),
                                  self.parent_record.primary_name,
                                  self.parent_record.primary_value,
                                  self.candidate_record.primary_value),
                configuration={
                    "epochs": self.epochs,
                    "learning_rate": float(self.candidate.model.meta["learning_rate"]),
                    "batch_size": int(self.candidate.model.meta["batch_size"]),
                    "hidden": self.candidate.hidden,
                    "seed": self.seed,
                    "parent_checkpoint_sha256": self.rnd.parent.checkpoint_sha256,
                    "hypothesis": self.hypothesis.key,
                    "applied": self.changed,
                    "validation_before": self.parent_record.values,
                    "validation_after": self.candidate_record.values,
                },
                seed=self.seed,
                checkpoint_paths=["checkpoint.npz"],
                code_paths=code_paths,
                predict_argv=["python3", "/app/harness.py", "predict"],
            )
            self.selected = submitted["artifact_id"]
        else:
            # Keep the parent.  The platform measures the parent, and the
            # round still has to end with a selection, so the parent's own
            # artefact is selected and the candidate is discarded.
            self.selected = self.rnd.parent.artifact_id
            shutil.rmtree(self.candidate_dir, ignore_errors=True)
            self.candidate_dir = None
        return self.selected

    def _copy_code(self, directory):
        """Copy the training code into the candidate as declared code paths.

        The rules ask for the code that produced the artefact to travel
        with it.  The image also carries the same code, so the artefact
        stays reproducible even if the image changes.
        """
        source = Path(__file__).resolve().parent.parent.parent
        target = directory / "code"
        target.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in ("contest_baselines",):
            origin = source / name
            if not origin.is_dir():
                continue
            shutil.copytree(origin, target / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            copied.append("code/%s" % name)
        return copied

    # --------------------------------------------------------------- finishing
    def finish(self):
        """Declare the round's outcome."""
        self.rnd.save_state(self.state_dir)
        self.rnd.finish(
            selected_artifact_id=self.selected,
            hypothesis=self.hypothesis.claim,
            change_summary="%s Applied: %s. Validation %s moved from %.4f to %.4f; adopted=%s."
                           % (self.hypothesis.change,
                              json.dumps(self.changed, ensure_ascii=False),
                              self.parent_record.primary_name,
                              self.parent_record.primary_value,
                              self.candidate_record.primary_value,
                              "yes" if self.adopted else "no"),
            reason="%s %s The model's review of this reading was: %s" % (
                self.reason,
                "The change was adopted."
                if self.adopted
                else "The change was not adopted, so the parent is kept.",
                self.review[:1500] if self.review else "(no text)",
            ),
            configuration={
                "epochs": self.epochs,
                "hypothesis": self.hypothesis.key,
                "adopted": self.adopted,
                "validation_before": self.parent_record.values,
                "validation_after": self.candidate_record.values,
                # The selection slice and the guard travel with the entry
                # so that the adoption decision can be reproduced from
                # the artefacts alone.  Without them a reader sees the
                # verdict but cannot check it - and the platform's own
                # report does not display the slice either.
                "selection_before": self.parent_record.selection_value,
                "selection_after": self.candidate_record.selection_value,
                "selection_metric": self.parent_record.selection_name,
                "selection_guard": self.guard,
                "model_call_id": self.receipt.call_id if self.receipt else "",
            },
            seed=self.seed,
            history_ids=[item["id"] for item in self.history],
            model_call_ids=list(self.rnd.call_ids),
        )
        log("round %d finished in %.1fs"
            % (self.rnd.round_index, time.monotonic() - self.started))

    # ---------------------------------------------------------------- the run
    def run(self):
        """Run the whole round, in order.

        The order is the contract, so it is written out explicitly rather
        than hidden behind the individual steps.
        """
        self.open_parent()
        self.read_public()
        self.plan()
        self.measure_parent()
        self.build_candidate()
        self.measure_candidate()
        self.decide()
        self.record()
        self.consult_the_model()
        self.submit()
        self.finish()
        return self


def _fmt(value):
    """Format a metric for a human-readable reason.

    The reason string goes into the ledger, the SwanLab record and the
    model's own review prompt, so it has to stay readable for a number
    that may be either very small (an accuracy) or very large (a
    remaining life in minutes).
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "nan"
    if abs(number) >= 1000.0:
        return "%.1f" % number
    if abs(number) >= 1.0:
        return "%.4f" % number
    return "%.6f" % number
