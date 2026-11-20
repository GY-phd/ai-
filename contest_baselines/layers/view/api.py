# -*- coding: utf-8 -*-
"""The View layer: a read-only API over the Model layer.

What a View is for
------------------
The competition's record of what happened has to be readable by
something other than this process.  A dashboard, a review page, a
spreadsheet export - all of them want the same thing: the ledger, the
measurements, the current champion, in a shape they can render.

So the View layer is an **API**: a set of named queries over the Model
layer's tables, returning plain dictionaries and lists that serialise
to JSON without any further thought.

Three rules
-----------
**Read-only.**  A View never writes.  Every query is a pure read of
    the Model layer, so a View can be pointed at a frozen archive and
    will give the same answer as it gives against a live directory.

**No knowledge of the protocol.**  A View does not know what a
    "round" is on the wire, does not import the SDK, and does not
    touch stdin or stdout.  It reads tables.

**No knowledge of the dataset.**  The View answers "what happened",
    not "what does A mean".  The profile's specifics live in the
    profile's adapter, and the View never asks which profile it is
    looking at - it reports what the tables say.

The shape of an answer
----------------------
Every endpoint returns a dictionary with a ``kind`` and a ``version``,
so a consumer can tell what it got without guessing::

    {"kind": "ledger", "version": 1, "entries": [...]}

The API surface
---------------
=========================  ===================================================
endpoint                   what it answers
=========================  ===================================================
``overview``               the headline: rounds, best score, champion
``ledger``                 every round's experiment record
``rounds``                 one record per round, in order
``metrics``                the measurements, per round
``champion``               the current best model and its measurement
``hypothesis``             the hypotheses and how each turned out
``predict``                one prediction, from the mounted artefact
=========================  ===================================================
"""
from __future__ import annotations

import json
from pathlib import Path

from ..model.metrics import MetricRecord
from ..model.state import RoundState

#: The version of the API's answer shapes.  A consumer that speaks
#: version 1 can rely on these keys.
API_VERSION = 1


def _answer(kind, **fields):
    """Wrap a payload in the API's envelope."""
    answer = {"kind": kind, "version": API_VERSION}
    answer.update(fields)
    return answer


class View:
    """A read-only API over one round's state.

    The View is constructed from a **directory**: the round's state
    directory, or the candidate directory, or the root of an archive.
    Whatever it is given, it reads and never writes.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.state = RoundState(self.directory)

    # ------------------------------------------------------------ the ledger
    def ledger(self):
        """Every round's experiment record, oldest first."""
        entries = self.state.read_ledger()
        return _answer("ledger", entries=[entry.as_record() for entry in entries])

    def rounds(self):
        """One record per round, in order."""
        entries = self.state.read_ledger()
        return _answer("rounds", rounds=[{
            "round": entry.round_index,
            "dataset_key": entry.dataset_key,
            "task": entry.task,
            "hypothesis": entry.hypothesis,
            "change": entry.change,
            "adopted": entry.adopted,
            "reason": entry.reason,
            "validation_before": entry.before,
            "validation_after": entry.after,
        } for entry in entries])

    # -------------------------------------------------------------- metrics
    def metrics(self):
        """The measurements, per round."""
        entries = self.state.read_ledger()
        rows = []
        for entry in entries:
            rows.append({
                "round": entry.round_index,
                "before": entry.before,
                "after": entry.after,
            })
        return _answer("metrics", metrics=rows)

    # ------------------------------------------------------------- overview
    def overview(self):
        """The headline numbers.

        The champion is the round with the best *after* measurement, and
        the comparison follows the task's direction - higher is better
        for a classification task, lower for a regression one.  A round
        that was not adopted is still reported, because a refuted
        hypothesis is part of the evidence.
        """
        entries = self.state.read_ledger()
        if not entries:
            return _answer("overview", rounds=0, champion=None, best=None)
        champion = None
        for entry in entries:
            if not entry.after:
                continue
            if champion is None:
                champion = entry
                continue
            if _better(entry, champion):
                champion = entry
        return _answer(
            "overview",
            rounds=len(entries),
            dataset_key=entries[-1].dataset_key,
            task=entries[-1].task,
            champion={
                "round": champion.round_index,
                "hypothesis": champion.hypothesis,
                "adopted": champion.adopted,
                "reason": champion.reason,
                "validation": champion.after,
            } if champion else None,
            best=_primary_of(champion) if champion else None,
        )

    def champion(self):
        """The current best model and its measurement."""
        answer = self.overview()
        return _answer("champion",
                       champion=answer.get("champion"),
                       best=answer.get("best"),
                       task=answer.get("task"),
                       dataset_key=answer.get("dataset_key"))

    # ---------------------------------------------------------- hypotheses
    def hypothesis(self):
        """The hypotheses and how each one turned out.

        This is the table a reviewer reads: it says what was tried, on
        which round, and whether it was confirmed or refuted.
        """
        entries = self.state.read_ledger()
        rows = []
        for entry in entries:
            rows.append({
                "round": entry.round_index,
                "hypothesis": entry.hypothesis,
                "change": entry.change,
                "adopted": entry.adopted,
                "reason": entry.reason,
                "validation_before": entry.before,
                "validation_after": entry.after,
            })
        return _answer("hypothesis", hypotheses=rows)

    # -------------------------------------------------------------- predict
    def predict(self, inputs, model_dir=None):
        """One prediction, from the mounted artefact.

        This is the View's only endpoint that touches the model, and it
        is a **read** of the artefact: it loads the checkpoint the
        platform mounted and answers one sample, exactly as the
        prediction path does.  It exists so that a dashboard can show
        what the model would say about a row, without running the
        platform's prediction loop.
        """
        from ...predict import predict_sample

        class _Sample:
            def __init__(self, task, payload):
                self.task = task
                self.inputs = payload
                self.context = None

        task = "regression"
        checkpoint = self._find_checkpoint(model_dir)
        if checkpoint is not None:
            from ...model import Model
            task = Model.load(checkpoint).meta["task"]
        result = predict_sample(_Sample(task, inputs), model_dir or self.directory)
        return _answer("predict", outputs=result.get("outputs"))

    def _find_checkpoint(self, model_dir=None):
        """The checkpoint inside the given directory, if there is one."""
        base = Path(model_dir) if model_dir else self.directory
        for name in ("checkpoint.npz", "model.npz"):
            candidate = base / name
            if candidate.is_file():
                return candidate
        return None

    # ------------------------------------------------------------ transport
    def to_json(self, answer):
        """The answer as JSON text.

        The API's transport is JSON, because JSON is what every web
        consumer already speaks and what the platform's own artefacts
        already use.
        """
        return json.dumps(answer, ensure_ascii=False, allow_nan=False,
                          indent=2, sort_keys=True)


def _primary_of(entry):
    """The entry's primary measurement, or ``None``."""
    if entry is None or not entry.after:
        return None
    values = entry.after
    for name in ("macro_f1", "rmse"):
        if name in values:
            return {"metric": name, "value": values[name]}
    return None


def _better(candidate, incumbent):
    """Is the candidate's measurement better than the incumbent's?"""
    new = _primary_of(candidate)
    old = _primary_of(incumbent)
    if new is None or old is None:
        return False
    if new["metric"] != old["metric"]:
        return False
    if new["metric"] == "macro_f1":
        return new["value"] > old["value"]
    return new["value"] < old["value"]
