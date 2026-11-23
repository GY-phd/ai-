# -*- coding: utf-8 -*-
"""Profile B's three protocols, re-measured after the lifetime fix.

The README quotes three numbers for profile B, each measured under a
different protocol:

* the **official split** - the three validation bearings the competition
  ships, with the model fitted on the nine official training bearings;
* **leave-one-bearing-out** - the nine training bearings, each held out
  in turn, which is the protocol the competition's "whole bearings are
  isolated" rule really describes;
* **a fresh condition** - a third grouping, held out by the operating
  condition rather than by the bearing's index.

This script measures the first two, and reports the official static
baseline on the same folds so every row is a same-conditions
comparison.  The third protocol is measured by
``research/tracks/track_b_final.py``.

Run::

    python tools/measure_profile_b_protocols.py
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

from tracks.hypotheses import CHAMPION_DEFAULTS  # noqa: E402
from tracks.profiles import get  # noqa: E402
from tracks.run import elapsed_counts  # noqa: E402


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=np.float64)
                                  - np.asarray(y_pred, dtype=np.float64)) ** 2)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("B")
    train = profile.load("train")
    validation = profile.load("validation")
    groups_all = np.asarray(train["groups"])
    order_all = np.asarray(train["order"])
    y_all = np.asarray(train["y"], dtype=np.float64)
    x_all = train["x"]

    recipe = dict(CHAMPION_DEFAULTS["B"])
    print("champion recipe: %s" % recipe)

    # -- the official split ------------------------------------------------
    model = profile.make_model(
        x_all, y_all, train["paths"], groups=groups_all, order=order_all,
        **recipe)
    elapsed = np.asarray(elapsed_counts(validation["groups"],
                                        validation["order"]), dtype=np.float64)
    prediction = model.predict(validation["x"], validation["groups"], elapsed)
    official = _rmse(validation["y"], prediction)
    print("\nOFFICIAL split (3 validation bearings): rmse %.2f" % official)

    # -- leave-one-bearing-out --------------------------------------------
    units = sorted(set(groups_all.tolist()))
    total = 0.0
    total_rows = 0
    per_unit = {}
    for held_out in units:
        keep = groups_all != held_out
        fold = profile.make_model(
            x_all[keep], y_all[keep], train["paths"],
            groups=groups_all[keep], order=order_all[keep], **recipe)
        held = ~keep
        elapsed = np.asarray(elapsed_counts(groups_all[held], order_all[held]),
                             dtype=np.float64)
        prediction = fold.predict(x_all[held], groups_all[held], elapsed)
        error = _rmse(y_all[held], prediction)
        per_unit[held_out] = error
        total += error ** 2 * held.sum()
        total_rows += held.sum()
        print("  held out %-12s rmse %8.2f" % (held_out, error), flush=True)
    lobo = (total / total_rows) ** 0.5
    print("LEAVE-ONE-BEARING-OUT (9 folds), pooled rmse %.2f" % lobo)

    payload = {
        "recipe": recipe,
        "official_split": official,
        "lobo_pooled": lobo,
        "lobo_per_unit": per_unit,
    }
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_protocols.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_protocols.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
