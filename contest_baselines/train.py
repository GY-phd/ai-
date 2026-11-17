# -*- coding: utf-8 -*-
"""The continuation driver: read the parent, run one experiment, write the evidence.

The round is a complete, self-contained unit:

1. **Read the parent.**  The parent checkpoint is opened through
   ``rnd.parent.path(...)``, which verifies the declared digest.  The
   parent is never copied, rewritten or searched for in ``/app``.

2. **Read the public data.**  The training and validation projections come
   from ``rnd.data``, using the paths the platform wrote into
   ``rnd.task["public_projection"]``.

3. **Choose one hypothesis.**  The choice depends on what the earlier
   rounds already tried, so the ten rounds form an experiment rather than
   ten repetitions.

4. **Run one experiment.**  The hypothesis changes the *training
   procedure* only; the model's forward pass and the checkpoint format are
   inherited from the parent.

5. **Measure and decide.**  The candidate is scored on the public
   validation set, and the score decides whether the candidate is
   submitted or the parent is kept.

6. **Write the evidence.**  A ledger entry records the hypothesis, its
   falsifier, the measurement, and the decision.  The ledger is saved as
   the round's native state, so the next round starts from the record of
   what has already been tried.

The decision rule is deliberately conservative: a candidate is adopted
only when it beats the parent on the public validation set *and* the
measurement is consistent with the hypothesis's stated expectation.  A
candidate that improves the score in a way the hypothesis did not predict
is treated as a refutation, not as a win.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .data import read_rows, unit_ids
from .metrics import is_better, primary_metric_name, primary_metric_value, score_metrics
from .model import Model, checkpoint_digest
from .strategies import build_plan, epoch_budget, prepare


def log(message):
    """Log to stderr.  stdout carries only the protocol JSONL."""
    print("[contest] %s" % message, file=os.sys.stderr, flush=True)


def workspace():
    """The writable directory for this round."""
    root = os.environ.get("CONTEST_WORK", "/workspace")
    return Path(root)


def _parent_checkpoint_name(rnd):
    """The parent checkpoint's path *within* the parent artifact.

    The parent declares its own files; the checkpoint is the one that
    looks like a checkpoint.  Nothing is searched for in ``/app`` and no
    hash is fabricated: the file is opened through ``rnd.parent.path``,
    which checks the digest the platform declared.
    """
    declared = list(rnd.parent.declared_paths())
    candidates = [name for name in declared if name.endswith(".npz")]
    if not candidates:
        candidates = [name for name in declared if "checkpoint" in name.lower()]
    if not candidates:
        raise RuntimeError("the parent artifact declares no checkpoint file")
    # The shortest name is the root-level one (``checkpoint.npz``).
    candidates.sort(key=lambda name: (name.count("/"), len(name)))
    return candidates[0]


def _parent_root(rnd):
    """Where the parent artefact is mounted.

    The platform mounts it at ``/previous``.  The offline test harness runs
    on a machine where that path does not exist, so it may point the
    mount at the local copy of the package through
    ``CONTEST_PARENT_ROOT``.  Nothing else changes: the file is still
    opened through ``rnd.parent.path``, which checks the digest the
    platform declared.
    """
    override = os.environ.get("CONTEST_PARENT_ROOT")
    if override:
        rnd.parent.root = Path(override)
    return rnd.parent.root


def _load_parent(rnd):
    """Open the parent checkpoint and check that it matches this task."""
    name = _parent_checkpoint_name(rnd)
    _parent_root(rnd)
    path = rnd.parent.path(name)
    model = Model.load(path)
    task = rnd.task.get("task")
    if task and model.meta.get("task") != task:
        raise RuntimeError("the parent checkpoint is %s but the task is %s"
                           % (model.meta.get("task"), task))
    if model.meta.get("paths") is None:
        raise RuntimeError("the parent checkpoint declares no feature schema")
    log("parent checkpoint %s: task=%s hidden=%d epochs=%d steps=%d features=%d"
        % (name, model.meta["task"], model.meta["hidden"],
           model.meta["epochs"], model.meta["steps"], len(model.meta["paths"])))
    return model, name, path


def _read_public(rnd, model):
    """Read the public training and validation projections."""
    task = model.meta["task"]
    classes = list(model.meta.get("classes") or [])
    train = read_rows(rnd.data.train_path, task, classes)
    validation = read_rows(rnd.data.validation_path, task, classes, train["paths"])
    if set(train["ids"]) & set(validation["ids"]):
        raise RuntimeError("the training and validation sample IDs overlap")
    log("public data: %d training rows, %d validation rows, %d features"
        % (len(train["ids"]), len(validation["ids"]), len(train["paths"])))
    return train, validation


def _measure(model, data, profile_key):
    """Score a model on a labelled split."""
    if model.meta["task"] == "classification":
        predicted = model.predict(data["x"]).argmax(axis=1)
    else:
        predicted = model.predict(data["x"])
    return score_metrics(model.meta["task"], data["y"], predicted, profile=profile_key)


def _write_checkpoint(model, directory):
    """Write the candidate checkpoint into a fresh directory."""
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / "checkpoint.npz"
    model.save(checkpoint)
    return checkpoint


def _copy_code(directory):
    """Copy the training code into the candidate as declared code paths.

    The rules ask for the code that produced the artefact to travel with
    the artefact.  The image also carries the same code, so the artefact
    stays reproducible even if the image changes.
    """
    code_dir = directory / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    names = []
    for source in sorted(here.glob("*.py")):
        shutil.copyfile(source, code_dir / source.name)
        names.append("code/" + source.name)
    return names


def _ledger_entry(rnd, hypothesis, changed, before, after, adopted, reason):
    """One line of the experiment ledger."""
    return {
        "round": rnd.round_index,
        "dataset_key": rnd.dataset_key,
        "task": rnd.task.get("task"),
        "hypothesis": hypothesis.key,
        "claim": hypothesis.claim,
        "change": hypothesis.change,
        "falsifier": hypothesis.falsifier,
        "expected_effect": hypothesis.expect,
        "applied": changed,
        "validation_before": before,
        "validation_after": after,
        "primary_metric": primary_metric_name(rnd.task.get("task", "regression")),
        "adopted": bool(adopted),
        "decision": reason,
        "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
        "stage_id": rnd.stage_id,
    }


def _append_ledger(state_dir, entry):
    """Append one entry to the ledger that travels to the next round."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def run_round(rnd):
    """Run one complete round.  Returns the submitted candidate's reply."""
    started = time.monotonic()
    profile_key = rnd.dataset_key
    task = rnd.task.get("task")

    model, parent_name, parent_path = _load_parent(rnd)
    train, validation = _read_public(rnd, model)

    history = rnd.history()
    hypothesis = build_plan(profile_key, rnd.round_index, history)
    log("round %d: hypothesis %r" % (rnd.round_index, hypothesis.key))

    parent_metrics = _measure(model, validation, profile_key)
    parent_primary = primary_metric_value(task, parent_metrics)
    log("parent %s = %.4f" % (primary_metric_name(task), parent_primary))

    # The candidate starts as an exact copy of the parent: the weights, the
    # Adam moments, the epoch counter and the normalisation state all carry
    # over.  Only the training procedure changes.
    candidate = Model.load(parent_path)

    seed = int(candidate.meta.get("seed", 42))
    x, y, changed = prepare(hypothesis, candidate, train, profile_key, seed)
    epochs = epoch_budget(hypothesis, rnd.round_index, rnd.wall_seconds)
    log("applying %r: %s" % (hypothesis.key, json.dumps(changed, ensure_ascii=False)))

    losses = candidate.fit(x, y, epochs)
    candidate_metrics = _measure(candidate, validation, profile_key)
    candidate_primary = primary_metric_value(task, candidate_metrics)
    log("candidate %s = %.4f (%d epochs, %.1fs)"
        % (primary_metric_name(task), candidate_primary, epochs,
           time.monotonic() - started))

    improved = is_better(task, candidate_primary, parent_primary)
    # The falsifier: an improvement that the hypothesis did not predict is
    # not accepted as a confirmation of the hypothesis.
    consistent = _consistent_with_expectation(task, hypothesis.key, improved,
                                              parent_primary, candidate_primary)
    adopted = bool(improved and consistent)
    if improved and not consistent:
        reason = ("the score improved (%s -> %s) but not in the direction the "
                  "hypothesis predicted, so the change is not adopted"
                  % (parent_primary, candidate_primary))
    elif improved:
        reason = ("the hypothesis is confirmed: %s moved from %s to %s"
                  % (primary_metric_name(task), parent_primary, candidate_primary))
    elif candidate_primary == parent_primary:
        reason = "the change left the score unchanged, so it is not adopted"
    else:
        reason = ("the hypothesis is refuted: %s moved from %s to %s"
                  % (primary_metric_name(task), parent_primary, candidate_primary))
    log("decision: %s" % reason)

    # ---- the candidate ----------------------------------------------------
    work = workspace()
    candidate_dir = work / ("candidate-round-%d" % rnd.round_index)
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    _write_checkpoint(candidate, candidate_dir)
    code_paths = _copy_code(candidate_dir)
    checkpoint_digest(candidate_dir / "checkpoint.npz")

    # ---- the ledger -------------------------------------------------------
    entry = _ledger_entry(rnd, hypothesis, changed, parent_metrics,
                           candidate_metrics, adopted, reason)
    state = rnd.state_workspace(str(work / "state"))
    _append_ledger(state, entry)

    # ---- the model call ---------------------------------------------------
    # The round must contain at least one real model call, and the call has
    # to be a genuine part of the decision rather than a formality: the
    # model is asked to check the reading of the measurement, and its reply
    # is recorded in the round's reason.
    receipt = rnd.model(
        [{"role": "user", "content": _model_brief(rnd, hypothesis, parent_primary,
                                                   candidate_primary, adopted)}],
        max_tokens=min(512, int(rnd.max_output_tokens)),
        temperature=0,
    )
    review = (receipt.text or "").strip()
    log("model call %s replied with %d characters" % (receipt.call_id, len(review)))

    # ---- the submission ---------------------------------------------------
    if adopted:
        submitted = rnd.submit_candidate(
            candidate_dir,
            hypothesis=hypothesis.claim,
            change_summary="%s Applied: %s. Validation %s moved from %.4f to %.4f."
                           % (hypothesis.change, json.dumps(changed, ensure_ascii=False),
                              primary_metric_name(task), parent_primary, candidate_primary),
            configuration={
                "epochs": epochs,
                "learning_rate": float(candidate.meta["learning_rate"]),
                "batch_size": int(candidate.meta["batch_size"]),
                "hidden": int(candidate.meta["hidden"]),
                "seed": seed,
                "parent_checkpoint_sha256": rnd.parent.checkpoint_sha256,
                "hypothesis": hypothesis.key,
                "applied": changed,
                "validation_before": parent_metrics,
                "validation_after": candidate_metrics,
            },
            seed=seed,
            checkpoint_paths=["checkpoint.npz"],
            code_paths=code_paths,
            predict_argv=["python3", "/app/harness.py", "predict"],
        )
        selected = submitted["artifact_id"]
        finish_reason = reason
    else:
        # Keep the parent.  The platform measures the parent, and the round
        # still has to end with a selection, so the parent's own artefact
        # is selected.
        selected = rnd.parent.artifact_id
        finish_reason = reason
        # A candidate that was not adopted must not be uploaded; the
        # directory is discarded so the round leaves no orphan.

    rnd.save_state(state)

    rnd.finish(
        selected_artifact_id=selected,
        hypothesis=hypothesis.claim,
        change_summary="%s Applied: %s. Validation %s moved from %.4f to %.4f; adopted=%s."
                       % (hypothesis.change, json.dumps(changed, ensure_ascii=False),
                          primary_metric_name(task), parent_primary, candidate_primary,
                          "yes" if adopted else "no"),
        reason="%s %s The model's review of this reading was: %s" % (
            finish_reason,
            "The change was adopted." if adopted
            else "The change was not adopted, so the parent is kept.",
            review[:1500] if review else "(no text)",
        ),
        configuration={
            "epochs": epochs,
            "hypothesis": hypothesis.key,
            "adopted": adopted,
            "validation_before": parent_metrics,
            "validation_after": candidate_metrics,
            "model_call_id": receipt.call_id,
        },
        seed=seed,
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )
    log("round %d finished in %.1fs" % (rnd.round_index, time.monotonic() - started))
    return selected


def _consistent_with_expectation(task, key, improved, before, after):
    """Does the measurement look like the hypothesis predicted?

    Every hypothesis in the bank predicts that the primary metric moves in
    one direction.  A change that moves the metric the *other* way cannot
    confirm the hypothesis, no matter how much the number moved, so it is
    reported as a refutation.
    """
    if not improved:
        return False
    if task == "classification":
        return after > before
    return after < before


def _model_brief(rnd, hypothesis, parent_primary, candidate_primary, adopted):
    """The message sent to the platform's model.

    Only this round's own state is sent: the hypothesis, the measurement
    and the decision.  The official prompt is **not** repeated here - the
    host attaches it to the request, and sending it again would duplicate
    it.
    """
    return (
        "本轮实验记录（数据集 %s，第 %d 轮，任务 %s）。\n"
        "假设：%s\n"
        "所做的改动：%s\n"
        "证伪条件：%s\n"
        "公开验证集上的主指标：父模型 %.4f -> 候选 %.4f。\n"
        "当前决策：%s。\n"
        "请只依据以上真实测量，判断这个读数是否支持该假设，并指出下一步最值得验证的一个改动。"
        % (rnd.dataset_key, rnd.round_index, rnd.task.get("task"),
           hypothesis.claim, hypothesis.change, hypothesis.falsifier,
           parent_primary, candidate_primary,
           "采纳候选" if adopted else "保留父模型")
    )
