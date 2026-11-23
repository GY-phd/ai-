# -*- coding: utf-8 -*-
"""Profile B: the multiplicative branch against the raw-minutes branch.

The hypothesis bank contains one experiment that turns the regression
**target** from the life fraction to the raw minutes.  After the
lifetime estimate was fixed, both branches became competitive:

    official validation bearings      life_fraction   raw minutes
    pooled RMSE                            445.78          408.16

so the champion cannot simply be the one the hypothesis claims.  This
script measures both branches under the protocol the competition's
"whole bearings are isolated" rule describes - **leave-one-bearing-out**
over the nine training bearings - and reports the pooled RMSE, which is
the metric profile B is actually scored on
(``tracks/task_metrics.py``).

It also reports the per-bearing numbers, because the two branches fail
differently: the two-stage branch has to extrapolate a lifetime for a
bearing it has never seen, and the raw-minutes branch does not.

Run::

    python tools/choose_profile_b_target.py
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
    units = sorted(set(groups_all.tolist()))
    base = dict(CHAMPION_DEFAULTS["B"])

    branches = [
        ("life_fraction (two-stage)", {}),
        ("minutes (raw target)", {"target": "minutes"}),
    ]

    print("=" * 96)
    print("leave-one-bearing-out, pooled RMSE (profile B's metric)")
    print("=" * 96)
    lobo = {}
    for label, over in branches:
        kwargs = dict(base)
        kwargs.update(over)
        total = 0.0
        total_rows = 0
        per_unit = {}
        for held_out in units:
            keep = groups_all != held_out
            model = profile.make_model(
                x_all[keep], y_all[keep], train["paths"],
                groups=groups_all[keep], order=order_all[keep], **kwargs)
            held = ~keep
            elapsed = np.asarray(elapsed_counts(groups_all[held], order_all[held]),
                                 dtype=np.float64)
            prediction = model.predict(x_all[held], groups_all[held], elapsed)
            error = _rmse(y_all[held], prediction)
            per_unit[held_out] = error
            total += error ** 2 * held.sum()
            total_rows += held.sum()
        pooled = (total / total_rows) ** 0.5
        lobo[label] = {"pooled": pooled, "per_unit": per_unit}
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-28s %10.2f   %s" % (label, pooled, detail), flush=True)

    print("\n" + "=" * 96)
    print("official validation bearings")
    print("=" * 96)
    official = {}
    for label, over in branches:
        kwargs = dict(base)
        kwargs.update(over)
        model = profile.make_model(
            x_all, y_all, train["paths"], groups=groups_all, order=order_all,
            **kwargs)
        elapsed = np.asarray(elapsed_counts(validation["groups"],
                                            validation["order"]), dtype=np.float64)
        prediction = model.predict(validation["x"], validation["groups"], elapsed)
        groups = np.asarray(validation["groups"])
        per_unit = {}
        for unit in sorted(set(groups.tolist())):
            mask = groups == unit
            per_unit[unit] = _rmse(np.asarray(validation["y"])[mask],
                                   prediction[mask])
        pooled = _rmse(validation["y"], prediction)
        official[label] = {"pooled": pooled, "per_unit": per_unit}
        detail = " ".join("%s=%.0f" % (k.replace("Bearing", "B"), v)
                          for k, v in sorted(per_unit.items()))
        print("%-28s %10.2f   %s" % (label, pooled, detail))

    print("\n" + "=" * 96)
    print("which branch fails where")
    print("=" * 96)
    for unit in units:
        left = lobo["life_fraction (two-stage)"]["per_unit"][unit]
        right = lobo["minutes (raw target)"]["per_unit"][unit]
        winner = "two-stage" if left < right else "minutes"
        print("  %-12s two-stage %8.1f   minutes %8.1f   -> %s"
              % (unit, left, right, winner))

    payload = {"lobo": lobo, "official": official}
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_b_target.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % (out / "profile_b_target.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
