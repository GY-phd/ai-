# -*- coding: utf-8 -*-
"""The ``Ledger`` table: the experiment trail, one round per line.

Why an append-only log
----------------------
The platform's state between rounds is a **directory of files with a
digest**.  It is copied into the next round, hashed, and the hash
travels with the round's declaration.  A JSONL log is the shape that
survives that journey:

* it is append-only, so a round never rewrites what an earlier round
  said - the trail cannot be edited into a different story;
* it is line-delimited, so one round's record can be read without
  parsing the whole file;
* it is plain text, so it is diffable and the platform can collect it
  without understanding it.

What a ledger entry says
------------------------
One entry per round, and it has to answer four questions:

**What was tried.**  The hypothesis, its claim, and its falsifier.  A
claim without a falsifier is a guess, and a round that records a guess
is not an experiment.

**What was changed.**  The concrete change to the *training procedure*:
the learning rate, the epoch budget, the training distribution, or the
hidden width.  The feature schema is not on that list, because it may
not change.

**What was measured.**  The metric before and after, on the public
validation set, under the parent's own conditions.

**What was decided, and why.**  Whether the candidate was adopted, and
the reason - including the case where the score improved but not in the
direction the hypothesis predicted, which is a *refutation* and not a
confirmation.
"""
from __future__ import annotations

import json
from pathlib import Path

#: The ledger's file name.  It travels in the round's state.
FILE_NAME = "ledger.jsonl"


class LedgerEntry:
    """One round's experiment record."""

    def __init__(self, round_index, dataset_key, task, hypothesis, change,
                 before, after, adopted, reason, **extra):
        self.round_index = int(round_index)
        self.dataset_key = dataset_key
        self.task = task
        self.hypothesis = hypothesis
        self.change = change
        self.before = dict(before or {})
        self.after = dict(after or {})
        self.adopted = bool(adopted)
        self.reason = reason
        self.extra = dict(extra)

    def as_record(self):
        """The JSON shape that is written to the ledger."""
        record = {
            "round": self.round_index,
            "dataset_key": self.dataset_key,
            "task": self.task,
            "hypothesis": self.hypothesis,
            "change": self.change,
            "validation_before": self.before,
            "validation_after": self.after,
            "adopted": self.adopted,
            "reason": self.reason,
        }
        record.update(self.extra)
        return record

    @classmethod
    def from_record(cls, record):
        """Read an entry back from its JSON shape."""
        known = ("round", "dataset_key", "task", "hypothesis", "change",
                 "validation_before", "validation_after", "adopted", "reason")
        extra = {key: value for key, value in record.items() if key not in known}
        return cls(
            record.get("round", 0),
            record.get("dataset_key"),
            record.get("task"),
            record.get("hypothesis"),
            record.get("change"),
            record.get("validation_before"),
            record.get("validation_after"),
            record.get("adopted"),
            record.get("reason"),
            **extra,
        )

    def __repr__(self):
        return "<LedgerEntry round=%d hypothesis=%r adopted=%s>" % (
            self.round_index, self.hypothesis, self.adopted)


class Ledger:
    """The append-only experiment trail."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.entries = []

    # ------------------------------------------------------------- writing
    def append(self, entry):
        """Add one entry and write it out.

        The write is **append-only**: the file is opened in ``"a"``
        mode, so an earlier round's line cannot be rewritten by a later
        one.  If the write fails the entry is still returned, because a
        telemetry failure must never cost a measured round - but the
        failure is raised, so the caller can see that the trail has a
        hole.
        """
        if isinstance(entry, dict):
            entry = LedgerEntry.from_record(entry)
        self.entries.append(entry)
        if self.path is None:
            return entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry.as_record(), ensure_ascii=False,
                                    allow_nan=False) + "\n")
        return entry

    # ------------------------------------------------------------- reading
    def read(self, path=None):
        """Read a ledger back.

        A malformed line is skipped rather than fatal: the ledger is
        evidence, and a reader that refuses to show the evidence
        because one line is broken is worse than a reader that shows
        everything it can.
        """
        target = Path(path) if path else self.path
        if target is None or not target.is_file():
            return self.entries
        self.entries = []
        with target.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    self.entries.append(LedgerEntry.from_record(record))
        return self.entries

    @classmethod
    def load(cls, path):
        """Open a ledger and read it."""
        ledger = cls(Path(path))
        ledger.read()
        return ledger

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)
