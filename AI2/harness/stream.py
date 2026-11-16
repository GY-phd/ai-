# -*- coding: utf-8 -*-
"""Build the evaluation stream by holding out whole bearings from training.

The defect this replaces
------------------------
The harness used to read the public validation file as its evaluation
stream.  That was wrong in two separate ways.

*At evaluation time there is no validation file.*  The platform supplies a
hidden stream; the public validation set exists only for local
development.  A harness whose stream path points at the validation file
cannot run on the hidden stream at all.

*It made every number optimistic.*  The public validation bearings are
``Bearing{1,2,3}_4`` - the fourth bearing of each working condition.  They
are continuations of conditions the initial model has seen, so a model
scores well on them partly because of that continuity rather than because
it generalised.  Measured on those bearings the official baseline scores
238 RMSE on bearings it was trained on and 681 on the validation bearings;
the gap is the "seen condition" advantage, not skill.

What this module does instead
-----------------------------
Hold out whole bearings from the **public training set** and use them as
the stream, exactly the way the rules describe the hidden split: "training,
validation and the hidden test stream are separated by whole bearings, and
snapshots of one bearing may not cross the split".

The holdout is **stratified by lifetime**.  Taking the longest bearings
would leave the initial training set without any examples of the large-RUL
regime, and the evaluation would then measure a scale mismatch instead of
model quality - the two would degrade together for a reason that has
nothing to do with the model.

The resulting layout:

``initial``
    The bearings the initial model is trained on.
``stream``
    The held-out bearings, concatenated and ordered by snapshot time.
    The harness sees them batch by batch, exactly like the hidden stream.
``validation``
    The public validation bearings, kept **completely untouched** by the
    harness.  They are scored once at the end as an independent check, so
    that the number reported for the slide is never the number the gate
    optimised against.

Run:
    python -m harness.stream --data D:\\GY\\cpu\\cpu-b-rul-20261110-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulkit.data import read_dataset  # noqa: E402


def _bearing_of(identity: str) -> str:
    """Recover the bearing name from a sample id such as ``..._s000123``."""
    return identity.rsplit("_s", 1)[0] if "_s" in identity else identity


def load_with_bearings(path: Path, expected_paths=None) -> dict:
    """Read a JSONL split and attach the bearing name to every row."""
    rows = read_dataset(path, expected_paths)
    rows["bearing"] = np.asarray([_bearing_of(i) for i in rows["ids"]])
    return rows


def stratified_holdout(bearings, n_holdout: int):
    """Choose bearings for the stream, stratified by lifetime.

    Bearings are ordered by lifetime (longest first) and then taken every
    ``stride``-th entry, so the holdout contains both short and long
    bearings.  This keeps the RUL range of the stream comparable to the
    RUL range of the initial training set.
    """
    ranked = sorted(bearings, key=lambda name: bearings[name], reverse=True)
    if not ranked:
        return []
    k = max(1, min(n_holdout, len(ranked)))
    stride = max(1, len(ranked) // k)
    picked = []
    for offset in range(stride):
        for name in ranked[offset::stride]:
            if len(picked) >= k:
                break
            if name not in picked:
                picked.append(name)
        if len(picked) >= k:
            break
    return picked


def build(data_root: Path, n_holdout: int = 3):
    """Split the public training set into an initial part and a stream part."""
    data_root = Path(data_root)
    train = load_with_bearings(data_root / "public" / "train.jsonl")
    validation = load_with_bearings(data_root / "public" / "validation.jsonl", train["paths"])

    train_bearings = sorted(set(train["bearing"]))
    overlap = set(train_bearings) & set(validation["bearing"])
    if overlap:
        raise ValueError("training and validation bearings overlap: %s" % sorted(overlap))

    counts = {name: int((train["bearing"] == name).sum()) for name in train_bearings}
    holdout = stratified_holdout(counts, n_holdout)

    is_stream = np.isin(train["bearing"], holdout)
    n_rows = len(train["x"])

    def _take(source, mask):
        if isinstance(source, np.ndarray) and source.shape[:1] == (n_rows,):
            return source[mask]
        if isinstance(source, list) and len(source) == n_rows:
            return [item for item, keep in zip(source, mask) if keep]
        return source

    initial = {k: _take(v, ~is_stream) for k, v in train.items()}
    stream = {k: _take(v, is_stream) for k, v in train.items()}

    # Order the stream by time within each bearing, and keep bearings in a
    # deterministic order.  The harness consumes it sequentially.
    order = np.lexsort((np.asarray(stream["ids"]), stream["bearing"]))
    for key, value in list(stream.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(stream["x"]),):
            stream[key] = value[order]
        elif isinstance(value, list) and len(value) == len(stream["x"]):
            stream[key] = [value[i] for i in order]

    meta = {
        "initial_bearings": [b for b in train_bearings if b not in holdout],
        "stream_bearings": holdout,
        "validation_bearings": sorted(set(validation["bearing"])),
        "initial_rows": int(len(initial["x"])),
        "stream_rows": int(len(stream["x"])),
        "validation_rows": int(len(validation["x"])),
        "initial_rul_max": float(initial["y"].max()),
        "stream_rul_max": float(stream["y"].max()),
        "initial_rul_median": float(np.median(initial["y"])),
        "stream_rul_median": float(np.median(stream["y"])),
        "bearing_leakage": False,
        "validation_used_for_training": False,
    }
    return initial, stream, validation, meta


def main(argv=None) -> int:
    """Print the stream split and its audit record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--holdout", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    initial, stream, validation, meta = build(Path(args.data), args.holdout)
    print("=== evaluation stream, built by holding out whole bearings ===")
    print("  initial training : %s" % ", ".join(meta["initial_bearings"]))
    print("                   %d rows, RUL 0..%.0f" % (meta["initial_rows"], meta["initial_rul_max"]))
    print("  stream           : %s" % ", ".join(meta["stream_bearings"]))
    print("                   %d rows, RUL 0..%.0f" % (meta["stream_rows"], meta["stream_rul_max"]))
    print("  validation       : %s" % ", ".join(meta["validation_bearings"]))
    print("                   %d rows (held out from every training and gate decision)" % meta["validation_rows"])
    print("\n  audit: bearing leakage %s | validation used for training %s" % (
        "NO" if not meta["bearing_leakage"] else "YES",
        "NO" if not meta["validation_used_for_training"] else "YES"))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
