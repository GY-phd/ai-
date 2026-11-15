# -*- coding: utf-8 -*-
"""The four-stage evidence chain: read history -> hypothesis -> experiment -> decision.

Why this module exists
----------------------
The competition asks for two things that a single final metric cannot show:

  ② "complete training / adjustment / evaluation records"
  ③ "at least 3 iterations of read history -> hypothesize -> experiment -> decide"

A metric trail alone cannot demonstrate either.  What a reviewer needs to see is
that the system *read* something, formed a *falsifiable* expectation, *ran* a
measurement, and then *decided* - including the cases where the decision was
"leave the champion alone".

Each :class:`HypothesisRecord` therefore carries all four stages explicitly.
The harness appends one record per cycle to ``hypothesis_chain.jsonl``; the
ledger hashes every record, so the chain cannot be edited after the fact.

This module was fused in from the ai+ repository, where the same four stages
were used to drive the training loop.  The datamodel is unchanged; only the
storage location and the hook into the orchestrator differ.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# The four stage keys, in order.  Kept as data so the report renderer and the
# self-check can both assert on them without duplicating the literal.
STAGES = ("read_history", "hypothesis", "experiment", "decision")


@dataclass
class HypothesisRecord:
    """One complete read -> hypothesize -> experiment -> decide cycle.

    Attributes
    ----------
    iteration:
        Monotonic cycle index, starting at 1.
    read_history:
        What the system observed before acting.  Typically the drift report,
        the diagnosis, and any earlier decisions that resemble this situation.
    hypothesis:
        A falsifiable statement.  It must name the expected direction *and* the
        condition under which it would be considered refuted.
    expected_gain:
        The RMSE improvement (in percent) the hypothesis predicts.
    falsification_criterion:
        Human-readable condition that would refute the hypothesis.
    experiment:
        What was actually run - the candidate configuration, its cost, and the
        measured outcome.
    decision:
        What the system did about it - promote / reject / do-nothing, with the
        reason.
    approved:
        Whether the decision was to promote the candidate.
    gain:
        The measured RMSE improvement (in percent); negative when it regressed.
    cost_gpu_seconds, cost_labels, cost_tokens:
        Resource accounting for this cycle.
    """

    iteration: int
    read_history: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    expected_gain: float = 0.0
    falsification_criterion: str = ""
    experiment: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    gain: float = 0.0
    cost_gpu_seconds: float = 0.0
    cost_labels: int = 0
    cost_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HypothesisRecord":
        """Rebuild a record from its serialised form."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def missing_stages(self) -> List[str]:
        """Return the stage names that carry no content.

        A record with an empty stage is not merely incomplete - it breaks the
        evidence chain the competition asks for.  The self-check reports these.
        """
        missing = []
        if not self.read_history:
            missing.append("read_history")
        if not self.hypothesis:
            missing.append("hypothesis")
        if not self.experiment:
            missing.append("experiment")
        if not self.decision:
            missing.append("decision")
        return missing

    def is_complete(self) -> bool:
        """True when all four stages are populated."""
        return not self.missing_stages()


class HypothesisChain:
    """Append-only store for :class:`HypothesisRecord` objects."""

    def __init__(self, workspace: Path):
        self.root = Path(workspace)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "hypothesis_chain.jsonl"
        self.records: List[HypothesisRecord] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.records.append(HypothesisRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        # A truncated trailing line must not make the whole
                        # chain unreadable; the ledger hash check still
                        # catches the tampering.
                        continue

    def append(self, record: HypothesisRecord) -> HypothesisRecord:
        """Append one record and persist it."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self.records.append(record)
        return record

    def next_iteration(self) -> int:
        """Return the index the next record should carry."""
        return len(self.records) + 1

    def summary(self) -> Dict[str, Any]:
        """Aggregate the chain into a report-friendly view."""
        complete = [r for r in self.records if r.is_complete()]
        promoted = [r for r in self.records if r.approved]
        rejected = [r for r in self.records if r.decision and not r.approved
                    and r.decision.get("action") != "do_nothing"]
        noop = [r for r in self.records
                if r.decision and r.decision.get("action") == "do_nothing"]
        gains = [r.gain for r in self.records if r.decision]
        return {
            "iterations": len(self.records),
            "complete_records": len(complete),
            "promoted": len(promoted),
            "rejected": len(rejected),
            "do_nothing": len(noop),
            "best_gain_pct": round(max(gains), 3) if gains else 0.0,
            "total_gain_pct": round(sum(gains), 3) if gains else 0.0,
            "missing_stage_records": [
                {"iteration": r.iteration, "missing": r.missing_stages()}
                for r in self.records if not r.is_complete()
            ],
        }
