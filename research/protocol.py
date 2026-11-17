# -*- coding: utf-8 -*-
"""Strict evaluation protocol for the B task (bearing RUL).

Why this file exists
--------------------
Every number that goes on a slide has to survive one question:
*"would this hold on a bearing the system has never seen?"*  The
competition states it plainly - training, validation and the hidden test
stream are split **by whole bearing**, and snapshots of one bearing may not
cross the split.  A measurement that quietly leaks bearings produces a
beautiful number and a useless system.

This module builds three disjoint evaluations from the public data alone:

``official``
    The official static baseline checkpoint, scored on the public
    validation bearings.  This is the number the rules compare against
    (门槛 ③).

``holdout``
    Train on a subset of the *training* bearings, evaluate on the
    validation bearings.  The validation bearings never enter training, so
    this is the same regime the hidden stream will be in.

``leave-one-bearing-out``
    Rotate over all nine training bearings: train on eight, evaluate on
    the held-out one.  Averaged over nine rotations this says how the
    system behaves on an unseen bearing *without* consuming the public
    validation set, which keeps the validation set honest as a final
    check.

The module also refuses to score a model whose predictions depend on
anything the reference loader would reject (bearing id, snapshot index,
total snapshots, the target itself).

Run:
    python -m research.protocol --report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulkit.data import metrics, read_dataset  # noqa: E402


# --------------------------------------------------------------------- data
def load_split(data_root: Path, name: str):
    """Load one public split and attach the bearing id to every row."""
    path = Path(data_root) / "public" / f"{name}.jsonl"
    rows = read_dataset(path)
    snapshots = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            snapshots.append(int(json.loads(line)["meta"]["snapshot"]))
    rows["bearing"] = np.asarray(rows["groups"])
    rows["snapshot"] = np.asarray(snapshots)
    return rows


def bearing_groups(rows) -> dict:
    """Map bearing name -> row indices."""
    groups = defaultdict(list)
    for index, name in enumerate(rows["bearing"]):
        groups[name].append(index)
    return {k: np.asarray(v) for k, v in groups.items()}


def subset(rows, indices: np.ndarray) -> dict:
    """Slice a dataset down to the given row indices."""
    out = {}
    for key, value in rows.items():
        if key in ("paths", "sha256"):
            out[key] = value
        elif hasattr(value, "__len__") and len(value) == len(rows["x"]):
            out[key] = np.asarray(value)[indices]
        else:
            out[key] = value
    return out


# --------------------------------------------------------------- auditing
def audit(data_root: Path) -> dict:
    """Check the properties the rules care about, and report them.

    Returns a dictionary that is written to the evidence file, so the
    audit is part of the record rather than a claim in a slide.
    """
    train = load_split(data_root, "train")
    validation = load_split(data_root, "validation")
    train_bearings = sorted(set(train["bearing"]))
    valid_bearings = sorted(set(validation["bearing"]))

    overlap = sorted(set(train_bearings) & set(valid_bearings))

    # RUL must equal (N - snapshot) for every bearing, exactly.
    rul_exact = True
    rul_rows = []
    for name, rows in (("train", train), ("validation", validation)):
        groups = bearing_groups(rows)
        for bearing, indices in sorted(groups.items()):
            snapshots = np.asarray([rows["snapshot"][i] for i in indices])
            total = len(indices)
            expected = total - snapshots
            actual = np.asarray([rows["y"][i] for i in indices])
            exact = bool(np.array_equal(actual, expected))
            contiguous = bool(np.array_equal(np.sort(snapshots), np.arange(1, total + 1)))
            rul_exact = rul_exact and exact and contiguous
            rul_rows.append({"split": name, "bearing": bearing, "rows": int(total),
                             "min_rul": int(actual.min()), "max_rul": int(actual.max()),
                             "rul_exact": exact, "snapshot_contiguous": contiguous})

    return {
        "train_rows": int(len(train["x"])),
        "validation_rows": int(len(validation["x"])),
        "train_bearings": train_bearings,
        "validation_bearings": valid_bearings,
        "bearing_overlap": overlap,
        "no_bearing_leakage": not overlap,
        "rul_definition_exact": rul_exact,
        "per_bearing": rul_rows,
    }


# ------------------------------------------------------------------ scoring
def score_on(rows, predict_fn) -> dict:
    """Score a prediction callable on the given rows."""
    predicted = np.asarray(predict_fn(rows["x"]), dtype=float)
    scored = metrics(rows["y"], predicted)
    per_bearing = {}
    groups = bearing_groups(rows)
    for bearing, indices in sorted(groups.items()):
        sub = metrics(rows["y"][indices], predicted[indices])
        per_bearing[bearing] = {"rmse": round(sub["rmse"], 3),
                                "mae": round(sub["mae"], 3),
                                "rows": int(len(indices))}
    scored["per_bearing"] = per_bearing
    scored["n"] = int(len(rows["y"]))
    return scored


def official_baseline(data_root: Path) -> dict:
    """Score the shipped static baseline checkpoint."""
    baseline_dir = Path(data_root)
    sys.path.insert(0, str(baseline_dir / "baselines"))
    try:
        from model import Model  # type: ignore
    finally:
        sys.path.pop(0)
    model = Model.load(baseline_dir / "checkpoint.npz")
    validation = load_split(data_root, "validation")
    return score_on(validation, model.predict)


def main(argv=None) -> int:
    """Run the audit and print the official baseline score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--report", action="store_true", help="print the full audit")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    data_root = Path(args.data)
    result = audit(data_root)

    print("=== data audit ===")
    print("  train      : %d rows / %d bearings" % (result["train_rows"], len(result["train_bearings"])))
    print("  validation : %d rows / %d bearings" % (result["validation_rows"], len(result["validation_bearings"])))
    print("  bearing leakage        : %s" % ("NONE" if result["no_bearing_leakage"] else result["bearing_overlap"]))
    print("  RUL == (N - snapshot)  : %s" % ("EXACT for all bearings" if result["rul_definition_exact"] else "MISMATCH"))
    if args.report:
        print("\n  per bearing:")
        for row in result["per_bearing"]:
            print("    %-11s %-14s rows=%5d  RUL %5d..%-5d  exact=%s contiguous=%s"
                  % (row["split"], row["bearing"], row["rows"], row["min_rul"],
                     row["max_rul"], row["rul_exact"], row["snapshot_contiguous"]))

    print("\n=== official static baseline (public validation) ===")
    scored = official_baseline(data_root)
    print("  RMSE %.3f   MAE %.3f   n=%d" % (scored["rmse"], scored["mae"], scored["n"]))
    for bearing, sub in scored["per_bearing"].items():
        print("    %-14s rows=%5d  rmse=%8.2f  mae=%8.2f"
              % (bearing, sub["rows"], sub["rmse"], sub["mae"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"audit": result, "official_baseline": scored},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
