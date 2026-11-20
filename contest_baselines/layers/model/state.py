# -*- coding: utf-8 -*-
"""The ``RoundState`` table: what travels to the next round.

What the state is for
---------------------
The platform hands the next round a **state directory**: the files the
previous round saved, mounted read-only, plus a digest that the next
round must carry forward unchanged.  It is the only channel between
rounds.

So the state has one job: to carry the evidence forward without
changing it.  It holds

* the **ledger** - every round's experiment record, so that the next
  round's plan can be a function of what has already been tried;
* the **SwanLab metric stream** - the experiment record in SwanLab's
  shape, so the platform collects a continuous log rather than a
  per-round fragment.

Why it is a table and not a helper
----------------------------------
Because the state is *written and read* rather than merely passed
along.  A round saves it, the platform hashes it, and the next round
reads it back.  Making that a single object with one writer and one
reader means the file names are stated once, and a round cannot
accidentally save a state that the next round will not look for.
"""
from __future__ import annotations

from pathlib import Path

from .ledger import FILE_NAME as LEDGER_FILE
from .ledger import Ledger


class RoundState:
    """The state a round saves and the next round inherits."""

    #: The files the state carries.
    LEDGER = LEDGER_FILE
    SWANLAB = "swanlab.jsonl"

    def __init__(self, directory):
        self.directory = Path(directory)

    # -------------------------------------------------------------- layout
    @property
    def ledger_path(self):
        return self.directory / self.LEDGER

    @property
    def swanlab_path(self):
        return self.directory / self.SWANLAB

    # ------------------------------------------------------------- writing
    def ledger(self):
        """The ledger, opened for appending."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return Ledger(self.ledger_path)

    def append_ledger(self, entry):
        """Add one round's record to the ledger."""
        return self.ledger().append(entry)

    def write_swanlab(self, source):
        """Copy the round's SwanLab record into the state.

        The record is *copied* rather than moved: the candidate
        directory keeps its own copy, and the state keeps one, so a
        round's evidence survives whether or not the candidate was
        adopted.
        """
        source = Path(source)
        if not source.is_file():
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.swanlab_path
        target.write_bytes(source.read_bytes())
        return target

    # ------------------------------------------------------------- reading
    def read_ledger(self):
        """The rounds recorded so far."""
        return Ledger.load(self.ledger_path).entries

    def tried_hypotheses(self):
        """The hypotheses the earlier rounds already tried.

        The plan is a function of this set: a hypothesis that has been
        tried is not tried again until the bank runs out.
        """
        return {entry.hypothesis for entry in self.read_ledger()
                if entry.hypothesis}

    def __repr__(self):
        return "<RoundState %s>" % self.directory
