# -*- coding: utf-8 -*-
"""Find where a profile's performance actually peaks.

Why this tool exists
--------------------
The closed loop can only adopt a candidate that beats the parent
checkpoint.  So the ceiling of the loop is the ceiling of the
**candidate generator**, and on Profile A the candidate generator was
measuring far below the parent: ten cycles produced one adoption and
nine candidates that scored 0.54-0.64 Macro-F1 against a parent at
0.72.

That is not a gate problem - a gate that rejects a worse model is
working.  It is a **search** problem: the hypothesis bank proposes a
handful of fixed overrides, and on Profile A none of them happened to
land on a better recipe.  The way to find out whether a better recipe
exists at all is to measure the landscape directly.

What it measures
----------------
For one profile, it fits the model over a grid of configurations and
reports each one's score **on the same judge set the gate uses** - the
held-back units the loop compares candidates on.  A configuration that
wins there is one the loop could actually adopt.

The judge set, not the validation set, is the right target: the
validation set is scored once at the very end, and a search that
peeks at it would be optimising against the final report.

The champion's current configuration is measured **in the same run**,
so the comparison is not across two pieces of code.

**No weights are written.**  The output is JSON and Markdown.

Run::

    python tools/sweep_profile.py --profile A
    python tools/sweep_profile.py --profile A --grid capacity
    python tools/sweep_profile.py --profile A --top 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.hypotheses import CHAMPION_DEFAULTS  # noqa: E402
from tracks.profiles import get  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402


# --------------------------------------------------------------------- grids
#
# Each grid is a list of keyword overrides.  They are deliberately
# small and hand-picked: a blind product of six knobs would be
# thousands of fits, and the point of this tool is to *measure*, not to
# burn a day of CPU.
GRIDS = {
    # The champion's recipe plus the two knobs the capacity sweep
    # already showed matter on Profile A.
    "capacity": [
        {"hidden": h, "epochs": e}
        for h in (16, 32, 64, 128)
        for e in (20, 50, 100, 200, 400)
    ],
    # The learning rate and the batch size, at the champion's shape.
    "optimiser": [
        {"hidden": 32, "epochs": 50, "learning_rate": lr,
         "batch_size": bs}
        for lr in (3e-4, 1e-3, 3e-3, 1e-2)
        for bs in (32, 64, 128, 256)
    ],
    # The activation and the feature transform - the two choices the
    # model code treats as first-class.
    "shape": [
        {"hidden": 32, "epochs": 50, "activation": act,
         "feature_transform": ft}
        for act in ("tanh", "relu")
        for ft in ("raw", "standard")
    ],
}


def _fit_and_score(profile, initial, judge, overrides):
    """Fit one configuration and score it on the judge set.

    Returns a dictionary with the score and the time it took, or a
    dictionary with ``error`` set when the fit did not produce a
    usable model.

    The scoring goes through the same conversion the loop uses: a
    classification model's ``predict`` returns the probability matrix,
    and the metric wants the predicted **labels**, so the argmax has
    to happen first.  Skipping that step is not a small mistake - the
    metric broadcasts a ``(rows,)`` vector against a ``(rows, classes)``
    one and fails.
    """
    started = time.time()
    try:
        model = profile.make_model(
            initial["x"], initial["y"], initial["paths"], **overrides)
        prediction = model.predict(judge["x"])
        if profile.task == "classification":
            prediction = np.asarray(prediction)
            if prediction.ndim == 2:
                prediction = prediction.argmax(axis=1)
        scores = profile.metric(judge["y"], prediction)
    except Exception as exc:  # a bad grid entry must not kill the sweep
        return {"error": "%s: %s" % (type(exc).__name__, exc),
                "seconds": time.time() - started}
    primary = "macro_f1" if profile.task == "classification" else "rmse"
    return {
        "primary_metric": primary,
        "primary_value": float(scores.get(primary, float("nan"))),
        "scores": {k: float(v) for k, v in scores.items()
                   if isinstance(v, (int, float))},
        "seconds": time.time() - started,
    }


def sweep(profile_key, grid_name, holdout, judge_units):
    """Measure the grid for one profile."""
    profile = get(profile_key)
    champion = dict(CHAMPION_DEFAULTS.get(profile_key, {}))

    initial, _stream, judge, _validation, meta = build_stream(
        profile_key, holdout, judge_units)

    if grid_name == "all":
        configurations = [dict(champion)]
        for name in ("capacity", "optimiser", "shape"):
            for item in GRIDS[name]:
                merged = dict(champion)
                merged.update(item)
                configurations.append(merged)
        # De-duplicate, keeping the first occurrence.
        seen = set()
        unique = []
        for item in configurations:
            key = json.dumps(item, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        configurations = unique
    else:
        configurations = []
        for item in GRIDS[grid_name]:
            merged = dict(champion)
            merged.update(item)
            configurations.append(merged)
        configurations.insert(0, champion)

    results = []
    for index, overrides in enumerate(configurations, 1):
        outcome = _fit_and_score(profile, initial, judge, overrides)
        outcome["overrides"] = overrides
        results.append(outcome)
        if "error" in outcome:
            print("  [%3d/%3d] %-56s ERROR %s"
                  % (index, len(configurations),
                     json.dumps(overrides, sort_keys=True), outcome["error"]),
                  flush=True)
        else:
            print("  [%3d/%3d] %-56s %s=%.6f"
                  % (index, len(configurations),
                     json.dumps(overrides, sort_keys=True),
                     outcome["primary_metric"], outcome["primary_value"]),
                  flush=True)
    return champion, results, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="A",
                        help="the profile to sweep (A, B, C)")
    parser.add_argument("--grid", default="all",
                        choices=sorted(GRIDS) + ["all"])
    parser.add_argument("--top", type=int, default=10,
                        help="how many of the best to report")
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--judge-units", type=int, default=2)
    parser.add_argument("--out", default=None,
                        help="where to write the JSON and Markdown")
    args = parser.parse_args(argv)

    profile = get(args.profile)
    print("sweeping profile %s (%s)" % (args.profile, profile.task),
          flush=True)

    champion, results, meta = sweep(args.profile, args.grid,
                                     args.holdout, args.judge_units)

    scored = [r for r in results if "error" not in r]
    if not scored:
        print("\nno configuration fitted successfully")
        return 1

    higher_is_better = profile.task == "classification"
    scored.sort(key=lambda r: r["primary_value"],
                reverse=higher_is_better)

    print("\n" + "=" * 78)
    print("the %d best of %d configurations (scored on the judge set)"
          % (min(args.top, len(scored)), len(scored)))
    print("=" * 78)
    print("%-9s %-56s %s" % ("champion?", "overrides", profile.task))
    champion_key = json.dumps(champion, sort_keys=True)
    for row in scored[:args.top]:
        key = json.dumps(row["overrides"], sort_keys=True)
        mark = "yes" if key == champion_key else ""
        print("%-9s %-56s %.6f"
              % (mark, key, row["primary_value"]))

    champion_rows = [r for r in scored
                     if json.dumps(r["overrides"], sort_keys=True) == champion_key]
    champion_value = champion_rows[0]["primary_value"] if champion_rows else None
    best = scored[0]
    if champion_value is not None:
        gain = (best["primary_value"] - champion_value
                if higher_is_better else champion_value - best["primary_value"])
        print("\nchampion    %s=%.6f" % (profile.task, champion_value))
        print("best        %s=%.6f" % (profile.task, best["primary_value"]))
        print("available   %s%.6f" % ("+" if gain >= 0 else "-", abs(gain)))
        if gain > 0:
            print("\nthe champion is NOT at the ceiling: %s is better."
                  % json.dumps(best["overrides"], sort_keys=True))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile": args.profile,
            "task": profile.task,
            "primary_metric": scored[0]["primary_metric"],
            "grid": args.grid,
            "holdout_units": args.holdout,
            "judge_units": args.judge_units,
            "meta": {k: v for k, v in meta.items()
                     if isinstance(v, (int, float, str, list))},
            "champion": champion,
            "champion_primary_value": champion_value,
            "results": results,
        }
        (out / "sweep.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        lines = ["# Profile %s sweep" % args.profile, "",
                 "primary metric: `%s` (higher is better: %s)"
                 % (scored[0]["primary_metric"], higher_is_better), "",
                 "scored on the judge set - the held-back units the loop "
                 "gates on.", "",
                 "| rank | overrides | %s |"
                 % scored[0]["primary_metric"],
                 "|---:|---|---:|"]
        for index, row in enumerate(scored, 1):
            lines.append("| %d | `%s` | %.6f |"
                         % (index,
                            json.dumps(row["overrides"], sort_keys=True),
                            row["primary_value"]))
        (out / "sweep.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")
        print("\nwrote %s" % (out / "sweep.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
