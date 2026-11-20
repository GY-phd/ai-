# -*- coding: utf-8 -*-
"""The ``Checkpoint`` table: the industrial model's persisted state.

The schema is **frozen by the platform**
----------------------------------------
The platform measures a candidate by running *its own* reference loader
against the hidden set.  That loader expects a ``contest.cpu-mlp.v1``
artefact: a specific set of arrays, a specific metadata block, and a
specific forward pass.  A checkpoint whose forward pass deviates from
the reference - even in a way that improves the public score - is scored
by the reference's rules, not by ours.

So this class does **not** design a format.  It owns the persistence of
the format the platform already froze, and its job is to make sure that
format is produced identically every time.

    30|The class wraps the reference implementation rather than re-implementing it.
The weights, the Adam moments, the normalisation state and the feature
schema are all held by the wrapped model; the wrapper owns where they
are written and how they are read back.

    40|What this layer guarantees
------------------------------
* a checkpoint that the reference loader accepts, or an error;
* a checkpoint whose feature schema is exactly the parent's, because the
  hidden rows carry the parent's schema and a mismatch scores nothing;
* a digest, so the platform's declared digest can be checked against
  the file that is actually on disk.
"""
from __future__ import annotations

from pathlib import Path

from ...model import FORMAT as _REFERENCE_FORMAT
from ...model import MAX_HIDDEN as _REFERENCE_MAX_HIDDEN
from ...model import Model as _ReferenceModel
from ...model import checkpoint_digest as _reference_digest

#: The artefact format the platform's reference loader opens.
FORMAT = _REFERENCE_FORMAT

#: The reference loader's ceiling on the hidden width.
MAX_HIDDEN = _REFERENCE_MAX_HIDDEN

#: The file name the platform's artefact declaration expects.
FILE_NAME = "checkpoint.npz"


class Checkpoint:
    """One persisted model state.

    ``Checkpoint`` is a thin, honest wrapper: it holds the reference
    model, writes it in the platform's format, reads it back, and
    reports its digest.  It deliberately adds no state of its own,
    because any state it added would have to survive a round-trip
    through the platform's artefact upload - and the platform only
    uploads the files it was told about.
    """

    def __init__(self, model):
        if not isinstance(model, _ReferenceModel):
            raise TypeError("a checkpoint wraps the reference model")
        self.model = model

    # -------------------------------------------------------------- metadata
    @property
    def task(self):
        return self.model.meta["task"]

    @property
    def classes(self):
        """The label list, for a classification checkpoint."""
        return list(self.model.meta.get("classes") or [])

    @property
    def paths(self):
        """The feature schema, in the platform's flattened order."""
        return list(self.model.meta["paths"])

    @property
    def hidden(self):
        return int(self.model.meta["hidden"])

    @property
    def epochs(self):
        return int(self.model.meta.get("epochs", 0))

    def configuration(self):
        """The training settings, for the round's declaration."""
        meta = self.model.meta
        return {
            "epochs": int(meta.get("epochs", 0)),
            "learning_rate": float(meta["learning_rate"]),
            "batch_size": int(meta["batch_size"]),
            "hidden": int(meta["hidden"]),
            "seed": int(meta.get("seed", 42)),
        }

    # ------------------------------------------------------------ persistence
    def save(self, directory):
        """Write the checkpoint into ``directory`` and return its path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FILE_NAME
        self.model.save(path)
        return path

    @classmethod
    def load(cls, path):
        """Read a checkpoint back."""
        return cls(_ReferenceModel.load(path))

    def digest(self, path):
        """The digest of the written file."""
        return checkpoint_digest(path)

    def migrate(self, hidden, seed=None):
        """Move the weights to a different hidden width.

        The migration is **explicit and evidence-preserving**: the first
        ``min(old, new)`` hidden units keep their incoming and outgoing
        weights, and the Adam moments travel with them.  So the result is
        a continuation of the parent and not a re-initialisation.

        The feature schema is untouched: a width change is a change to
        the *model*, never to the *data*, and the platform's hidden rows
        carry the parent's schema.
        """
        note = self.model.resize(hidden, seed=seed)
        return note

    def __repr__(self):
        return "<Checkpoint task=%s hidden=%d epochs=%d>" % (
            self.task, self.hidden, self.epochs)


def checkpoint_digest(path):
    """The digest of a checkpoint file, in the platform's format."""
    return _reference_digest(path)
