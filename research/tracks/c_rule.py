# -*- coding: utf-8 -*-
"""Offline replay of the conjunction rule on the platform's own round records.

The problem
-----------
Profile C's runtime adopts a candidate only when the **full split** does
not regress *and* the protocol-aligned slice improves.  On the platform's
2026-11-23 run that rule rejected several candidates whose full split
improved:

======  ==========  ==========  ========
round   full rmse   late rmse   verdict
======  ==========  ==========  ========
2       52.23->51.18  38.52->38.84  conflict
7       50.91->50.88  36.21->36.53  conflict
======  ==========  ==========  ========

The final model was the round 5 candidate (full 51.111), not the round 7
one (full 50.878, the best full-split score of the whole run).

Three candidate rules are compared here:

``strict`` (today)
    Adopt only when the full split does not regress and the slice
    improves.

``tolerance``
    Adopt when the full split improves and the slice's regression stays
    inside a tolerance.

``weighted``
    Combine the two readings into one score with a weight and adopt when
    the score improves.

The question is which rule picks the candidate that actually wins on the
**hidden** split.  The only honest answer has to come from a held-out
evaluation, not from the public split - so this module replays the rule
over a set of trained candidates and scores the resulting selection under
a protocol that does not see the public validation split.

Run::

    python -m research.tracks.c_rule --folds 4 --epochs 30 --seeds 42
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tracks.c_holdout import (  # noqa: E402
    baseline_on, engine_folds, evaluate, train_on,
)
from research.tracks.c_protocol import load, rmse  # noqa: E402


# ------------------------------------------------------------------ rules
def rule_strict(full_before, full_after, slice_before, slice_after):
    """Today's rule: the full split must not regress and the slice must improve."""
    if full_after > full_before:
        return False
    return slice_after < slice_before


def rule_tolerance(tol):
    """Adopt when the full split improves and the slice stays inside a tolerance."""
    def decide(fb, fa, sb, sa):
        if fa >= fb:
            return False
        if sa <= sb:
            return True
        return (sa - sb) / sb <= tol
    return decide


def rule_weighted(weight):
    """Combine the two readings into one score; adopt when the score improves.

    Both readings are normalised by their parent value so the two scales
    are comparable, then combined with ``weight`` on the slice.
    """
    def decide(fb, fa, sb, sa):
        rel_full = (fa - fb) / fb if fb > 0 else 0.0
        rel_slice = (sa - sb) / sb if sb > 0 else 0.0
        score_before = 0.0
        score_after = weight * rel_slice + (1.0 - weight) * rel_full
        return score_after < score_before
    return decide


def rule_guarded(tol, guard):
    """Tolerance on the slice, plus a hard cap on the full-split regression."""
    def decide(fb, fa, sb, sa):
        rel_full = (fa - fb) / fb if fb > 0 else 0.0
        if rel_full > guard:
            return False
        if sa <= sb:
            return True
        return (sa - sb) / sb <= tol
    return decide


RULES = {
    "strict": rule_strict,
    "tol0.5%": rule_tolerance(0.005),
    "tol1%": rule_tolerance(0.01),
    "tol2%": rule_tolerance(0.02),
    "tol5%": rule_tolerance(0.05),
    "w0.25": rule_weighted(0.25),
    "w0.5": rule_weighted(0.5),
    "w0.75": rule_weighted(0.75),
    "tol1%+guard5%": rule_guarded(0.01, 0.05),
    "tol2%+guard5%": rule_guarded(0.02, 0.05),
}


# ------------------------------------------------------- the candidate pool
def candidates():
    """The recipes the closed loop actually produced."""
    return [
        ("control", dict(w="none")),
        ("surv150", dict(w="surv", H=150, scale=75, floor=0.05)),
        ("surv100", dict(w="surv", H=100, scale=50, floor=0.05)),
        ("surv195", dict(w="surv", H=195, scale=75, floor=0.05)),
        ("deg0.5", dict(w="deg", alpha=0.5)),
        ("deg1", dict(w="deg", alpha=1.0)),
        ("deg2", dict(w="deg", alpha=2.0)),
        ("deg2p0.5", dict(w="deg", alpha=2.0, power=0.5)),
        ("unit", dict(w="unit")),
        ("cap150", dict(w="none", cap=150)),
        ("cap200", dict(w="none", cap=200)),
    ]


def measure(ctx, model, rows_mask):
    """The two readings the runtime sees, on the training-side split."""
    y = ctx["train"]["y"][rows_mask]
    p = model.predict(ctx["train"]["x"][rows_mask])
    full = rmse(y, p)
    le = y <= 200
    late = rmse(y[le], p[le]) if le.any() else float("nan")
    return full, late


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_rule.json"))
    args = parser.parse_args(argv)

    ctx = load(args.package or None)
    folds = engine_folds(ctx, args.folds)
    seeds = [int(s) for s in args.seeds.split(",")]
    print("public train engines %d -> %d folds; %d candidates; %d rules"
          % (len(ctx["t_engines"]), args.folds, len(candidates()), len(RULES)), flush=True)

    records = []
    for seed in seeds:
        for fold_index, fold in enumerate(folds):
            held = np.isin(ctx["tunit"], fold)
            train_mask = ~held
            base = baseline_on(ctx, held)
            pool = []
            for name, spec in candidates():
                started = time.time()
                model = train_on(ctx, spec, train_mask, epochs=args.epochs,
                                 lr=0.004, batch=256, seed=seed)
                full, late = measure(ctx, model, train_mask)
                held_scores = evaluate(ctx, model, held)
                pool.append({
                    "name": name, "spec": spec,
                    "read_full": full, "read_late": late,
                    "holdout_full": held_scores["holdout_full_rmse"],
                    "holdout_late200": held_scores["holdout_late200_rmse"],
                    "holdout_mae": held_scores["holdout_mae"],
                    "seconds": round(time.time() - started, 1),
                })
                print("  fold%d %-11s read full %.4f late %.4f | holdout full %.4f late200 %.4f"
                      % (fold_index, name, full, late, held_scores["holdout_full_rmse"],
                         held_scores["holdout_late200_rmse"]), flush=True)

            # The parent is the official checkpoint, measured the same way.
            parent_model = train_on(ctx, dict(w="none"), train_mask, epochs=0,
                                    lr=0.004, batch=256, seed=seed)
            p_full, p_late = measure(ctx, parent_model, train_mask)
            parent = {"name": "parent", "read_full": p_full, "read_late": p_late,
                      "holdout_full": base["holdout_full_rmse"],
                      "holdout_late200": base["holdout_late200_rmse"],
                      "holdout_mae": base["holdout_mae"]}

            # Each rule walks the pool the way the runtime does: one round
            # at a time, the parent advances when a candidate is adopted.
            for rule_name, rule in RULES.items():
                current = dict(parent)
                trace = []
                for candidate in pool:
                    adopted = rule(current["read_full"], candidate["read_full"],
                                   current["read_late"], candidate["read_late"])
                    trace.append({
                        "candidate": candidate["name"], "adopted": bool(adopted),
                        "read_full": candidate["read_full"],
                        "read_late": candidate["read_late"],
                    })
                    if adopted:
                        current = candidate
                records.append({
                    "seed": seed, "fold": fold_index, "rule": rule_name,
                    "selected": current["name"],
                    "holdout_full": current["holdout_full"],
                    "holdout_late200": current["holdout_late200"],
                    "holdout_mae": current["holdout_mae"],
                    "parent_holdout_full": parent["holdout_full"],
                    "parent_holdout_late200": parent["holdout_late200"],
                    "trace": trace,
                })
                print("    fold%d %-16s -> %-11s holdout full %.4f (parent %.4f)"
                      % (fold_index, rule_name, current["name"], current["holdout_full"],
                         parent["holdout_full"]), flush=True)

    # Aggregate: a rule is good if its selection beats the parent on the
    # held-out engines, and if it beats the other rules.
    print("\n=== 汇总：各规则选出的候选在留出引擎上的表现 ===")
    summary = {}
    for rule_name in RULES:
        rows = [r for r in records if r["rule"] == rule_name]
        wins = sum(1 for r in rows if r["holdout_full"] < r["parent_holdout_full"] - 1e-9)
        wins_late = sum(1 for r in rows if r["holdout_late200"] < r["parent_holdout_late200"] - 1e-9)
        summary[rule_name] = {
            "folds": len(rows),
            "mean_holdout_full": float(np.mean([r["holdout_full"] for r in rows])),
            "mean_holdout_late200": float(np.mean([r["holdout_late200"] for r in rows])),
            "parent_mean_holdout_full": float(np.mean([r["parent_holdout_full"] for r in rows])),
            "parent_mean_holdout_late200": float(np.mean([r["parent_holdout_late200"] for r in rows])),
            "beats_parent_full": wins,
            "beats_parent_late200": wins_late,
            "selected": {},
        }
        for r in rows:
            summary[rule_name]["selected"][r["selected"]] = \
                summary[rule_name]["selected"].get(r["selected"], 0) + 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records, "summary": summary},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for rule_name, s in sorted(summary.items(),
                               key=lambda kv: kv[1]["mean_holdout_full"]):
        print("  %-16s holdout full %8.4f (parent %8.4f, win %d/%d)  late200 %8.4f (win %d/%d)"
              % (rule_name, s["mean_holdout_full"], s["parent_mean_holdout_full"],
                 s["beats_parent_full"], s["folds"],
                 s["mean_holdout_late200"], s["beats_parent_late200"], s["folds"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
