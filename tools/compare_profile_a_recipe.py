# -*- coding: utf-8 -*-
"""Compare the old and the new Profile A recipe, end to end.

The closed loop's final score comes from a **refit** on every row seen
so far - the initial batch plus the stream.  That refit trains on 755
rows across two batches, so the model sees the drifted data, and the
class whose signature is reversed between the batches (class 3,
Acetaldehyde) becomes learnable.

The question this script answers is whether the new recipe is actually
better **on the validation set the loop never trains on** - the
untouched official batch.  It runs the same loop twice, once with the
old recipe and once with the new one, and compares the final scores.

It reports the per-class F1 as well as the headline, because the whole
story is in class 3.

Run::

    python tools/compare_profile_a_recipe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.hypotheses import CHAMPION_DEFAULTS  # noqa: E402
from tracks.profiles import CLASSES_A  # noqa: E402

OLD_RECIPE = {"hidden": 32, "epochs": 50}
NEW_RECIPE = {"hidden": 32, "epochs": 50, "feature_transform": "slog",
              "learning_rate": 0.03, "batch_size": 256}


def _run_loop(recipe, workspace, cycles, holdout, judge):
    """Run the closed loop with one recipe and return its report."""
    from tracks import run as runner

    original = dict(CHAMPION_DEFAULTS["A"])
    CHAMPION_DEFAULTS["A"].clear()
    CHAMPION_DEFAULTS["A"].update(recipe)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            runner.main(["--profile", "A", "--workspace", workspace,
                         "--holdout-units", str(holdout),
                         "--judge-units", str(judge)])
        report = json.loads(
            (Path(workspace) / "A" / "run_report.json").read_text(
                encoding="utf-8"))
        return report
    finally:
        CHAMPION_DEFAULTS["A"].clear()
        CHAMPION_DEFAULTS["A"].update(original)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="runs/compare-a")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--judge", type=int, default=2)
    args = parser.parse_args(argv)

    results = {}
    for name, recipe in (("old", OLD_RECIPE), ("new", NEW_RECIPE)):
        workspace = str(Path(args.workspace) / name)
        print("\n" + "=" * 74)
        print("recipe: %s  %s" % (name, json.dumps(recipe, sort_keys=True)))
        print("=" * 74, flush=True)
        report = _run_loop(recipe, workspace, args.cycles,
                           args.holdout, args.judge)
        final = report["final_score"]
        initial = report["initial_score"]
        results[name] = {
            "recipe": recipe,
            "initial_macro_f1": initial["macro_f1"],
            "final_macro_f1": final["macro_f1"],
            "final_accuracy": final["accuracy"],
            "promotions": report["promotions"],
            "per_class": {CLASSES_A[i]: final[str(i)]
                          for i in range(len(CLASSES_A))},
        }

    print("\n" + "=" * 74)
    print("the two recipes, on the validation set the loop never trains on")
    print("=" * 74)
    print("%-8s %-12s %-12s %-10s" % ("recipe", "initial", "final", "promotions"))
    for name in ("old", "new"):
        row = results[name]
        print("%-8s %-12.4f %-12.4f %-10d"
              % (name, row["initial_macro_f1"], row["final_macro_f1"],
                 row["promotions"]))

    print("\nper-class F1 on that validation set:")
    print("  %-8s %s" % ("recipe",
                         " ".join("%-14s" % name[:13] for name in CLASSES_A)))
    for name in ("old", "new"):
        print("  %-8s %s"
              % (name, " ".join("%-14.3f" % results[name]["per_class"][c]
                                for c in CLASSES_A)))

    old = results["old"]["final_macro_f1"]
    new = results["new"]["final_macro_f1"]
    print("\nchange: %+.4f Macro-F1" % (new - old))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
