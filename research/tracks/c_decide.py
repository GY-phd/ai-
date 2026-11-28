# -*- coding: utf-8 -*-
"""The two decisions, measured on held-out engines, fold by fold.

Two questions the production code has to answer, and the only honest
protocol for both is the **leave-one-engine-out** one: train on some
engines, score a *different* engine group the model has never seen.  The
hidden C pack is a different dataset, so the public split's own engines
are the only stand-in for it.

Question 1 - which selection rule?
    The platform's 2026-11-25 run rejected candidates whose full split
    improved (rounds 6, 8 and 9) and then spent the remaining five
    rounds retaining the parent.  ``research/tracks/c_rule.py`` compares
    the rules; this module is the *decision* half of that comparison -
    it scores the rules on the same folds, one rule at a time, and
    reports the **paired** per-fold difference against the rule that
    selects on the full split alone.

Question 2 - which hyper-parameters?
    ``research/tracks/c_tune.py`` swept 24 recipes and found the best
    average, but an average over recipes does not say whether the
    winner is *reliably* better than the shipped configuration.  Here
    every recipe is scored on the **same** folds and the difference to
    the shipped configuration is reported fold by fold, with a paired
    Wilcoxon test over those folds.

The protocol
------------
For each ``(seed, fold)``:

1. train the shipped configuration and every candidate configuration on
   the training engines (``folds - 1`` groups);
2. score each on the held-out engines, and on the shipped static
   checkpoint under the identical mask - the baseline is the ruler,
   because the platform's gate is "not below the static baseline";
3. score each candidate under every selection rule, by replaying the
   runtime's own walk over the candidate pool: the parent advances only
    4|   when the rule adopts, so the rule under test is the *only* thing
   that differs between the arms.

Everything is reported per fold, so the numbers can be checked against
the fold they came from.

Run::

    python -m research.tracks.c_decide --folds 6 --seeds 42,7 --epochs 80
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


# ------------------------------------------------------------------ the rules
def rule_full_only(fb, fa, sb, sa):
    """Adopt when the **full split** improves.  The slice is not read."""
    return fa < fb


def rule_full_and_slice(fb, fa, sb, sa):
    """The conjunction the platform deployed: the full split must not
    regress *and* the slice must improve."""
    if fa >= fb:
        return False
    return sa < sb


def rule_slice_only(fb, fa, sb, sa):
    """The rule the evidence says is wrong: adopt on the slice alone."""
    return sa < sb


def rule_tolerance(tol):
    """Adopt when the full split improves and the slice stays within a tolerance."""
    def decide(fb, fa, sb, sa):
        if fa >= fb:
            return False
        if sa <= sb:
            return True
        return (sa - sb) / sb <= tol
    return decide


RULES = {
    "full-only": rule_full_only,
    "full+slice": rule_full_and_slice,
    "slice-only": rule_slice_only,
    "tol1%": rule_tolerance(0.01),
    "tol2%": rule_tolerance(0.02),
}


# --------------------------------------------------------------- the recipes
def recipes():
    """The configurations under test, each ``(name, spec, epochs, lr, hidden)``.

    The candidates come from the 24-recipe sweep in
    ``research/tracks/c_tune.py`` (two seeds, six engine folds, holdout
    RMSE against the fold's baseline):

    ===================  ==========  ==============
    recipe               avg delta   wins
    ===================  ==========  ==============
    ``ep80_lr0.008``     -0.9422     11 / 12
    ``ep120``            -0.7829     10 / 12
    ``ep100``            -0.6903      9 / 12
    ``alpha0.25``        -0.6659     12 / 12
    ``ep80``             -0.6546     11 / 12
    ``shipped``          -0.6237     10 / 12
    ===================  ==========  ==============

    ``shipped`` is the control: ``deg0.5`` at 40 epochs, lr 0.004 and a
    hidden width of 32.

    One more arm is here because of a **production/study mismatch** that
    the sweep could not see: the sweep's ``lr=0.004`` is not the rate the
    production loop actually trains at.  The shipped C checkpoint declares
    ``learning_rate = 0.002`` (see
    ``art/repo-inspect/cpu-c-guide-20261110-v1/checkpoint.npz``), and
    ``contest_baselines.strategies.prepare`` inherits it - only the
    ``the-parent-was-left-unconverged`` hypothesis doubles it.  So the
    study's control arm is *not* what the platform ran.  ``prod-lr``
    measures the rate production actually uses, at the epoch budget
    production actually uses, so the comparison is against the thing
    that ships and not against a proxy of it.
    """
    deg = dict(w="deg", alpha=0.5)
    return [
        ("shipped", deg, 40, 0.004, 32),
        ("prod-lr", deg, 40, 0.002, 32),
        ("ep80", deg, 80, 0.004, 32),
        ("ep120", deg, 120, 0.004, 32),
        ("ep80_lr0.008", deg, 80, 0.008, 32),
        ("ep80_alpha0.25", dict(w="deg", alpha=0.25), 80, 0.004, 32),
        ("ep80_alpha0.25_lr0.008", dict(w="deg", alpha=0.25), 80, 0.008, 32),
        ("ep120_alpha0.25", dict(w="deg", alpha=0.25), 120, 0.004, 32),
    ]


def wilcoxon(a, b):
    """A paired Wilcoxon signed-rank test on two per-fold vectors.

    Returns ``(statistic, p_value, n)``.  The normal approximation is
    used with the continuity correction; the exact test is not needed
    here because the number of folds is small enough that the p-values
    are only ever read as "clearly smaller than 0.05" or not.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = np.argsort(np.argsort(np.abs(d))).astype(float) + 1.0
    # average ranks for ties
    order = np.argsort(np.abs(d))
    sorted_abs = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    stat = min(r_plus, r_minus)
    if sigma <= 0:
        return stat, 1.0, n
    z = (stat - mu + 0.5) / sigma
    # two-sided normal tail
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return stat, min(1.0, p), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=80,
                    help="the epoch budget the *rules* are replayed at")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_decide.json"))
    args = ap.parse_args(argv)

    ctx = load(args.package or None)
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    print("public train engines %d -> %d folds; %d recipes; %d rules"
          % (len(ctx["t_engines"]), args.folds, len(recipes()), len(RULES)),
          flush=True)

    records = []
    for seed in seeds:
        folds = engine_folds(ctx, args.folds, seed=seed)
        for fold_index, held in enumerate(folds):
            held_mask = np.isin(ctx["tunit"], held)
            train_mask = ~held_mask
            base = baseline_on(ctx, held_mask)
            # The parent the rules start from: the shipped static
            # checkpoint, measured exactly the way the runtime measures
            # the parent.
            #
            # The readings the rules consume come from the **public
            # validation split**, because that is where the runtime
            # reads them (``RoundController._measure`` is called with
            # ``self.validation``).  ``research/tracks/c_rule.py``
            # instead reads the training-side split; the difference
            # matters, because the runtime's decision is made on data
            # the candidate was *not* trained on, and a rule that only
            # ever sees in-sample readings is not the rule that ships.
            # The scoring, in contrast, stays on the held-out engines:
            # that is the only reading that stands in for the hidden
            # pack.
            parent_model = train_on(ctx, dict(w="none"), train_mask,
                                    epochs=0, lr=0.004, batch=256, seed=seed)
            vy = ctx["valid"]["y"]
            vp = parent_model.predict(ctx["valid"]["x"])
            parent = {
                "name": "parent",
                "read_full": rmse(vy, vp),
                "read_late": rmse(vy[vy <= 200], vp[vy <= 200]),
                "holdout_full": base["holdout_full_rmse"],
            }

            pool = []
            for name, spec, epochs, lr, hidden in recipes():
                t0 = time.time()
                model = train_on(ctx, spec, train_mask, epochs=epochs, lr=lr,
                                 batch=256, seed=seed, hidden=hidden)
                rp = model.predict(ctx["valid"]["x"])
                late = rmse(vy[vy <= 200], rp[vy <= 200])
                scores = evaluate(ctx, model, held_mask)
                pool.append({
                    "name": name, "spec": spec, "epochs": epochs, "lr": lr,
                    "hidden": hidden,
                    "read_full": rmse(vy, rp), "read_late": late,
                    "holdout_full": scores["holdout_full_rmse"],
                    "seconds": round(time.time() - t0, 1),
                })
                print("  s%d f%d %-22s read %8.4f / %8.4f | holdout %8.4f  %.0fs"
                      % (seed, fold_index, name, pool[-1]["read_full"],
                         pool[-1]["read_late"], pool[-1]["holdout_full"],
                         pool[-1]["seconds"]), flush=True)

            # ---- the rules, replayed over the same pool ----------------
            for rule_name, rule in RULES.items():
                current = dict(parent)
                trace = []
                for candidate in pool:
                    adopted = rule(current["read_full"], candidate["read_full"],
                                   current["read_late"], candidate["read_late"])
                    trace.append({"candidate": candidate["name"],
                                  "adopted": bool(adopted)})
                    if adopted:
                        current = candidate
                records.append({
                    "seed": seed, "fold": fold_index, "rule": rule_name,
                    "selected": current["name"],
                    "holdout_full": current["holdout_full"],
                    "parent_holdout_full": parent["holdout_full"],
                    "delta": current["holdout_full"] - parent["holdout_full"],
                    "trace": trace,
                })
                print("    s%d f%d %-12s -> %-22s holdout %8.4f (parent %8.4f, %+0.4f)"
                      % (seed, fold_index, rule_name, current["name"],
                         current["holdout_full"], parent["holdout_full"],
                         current["holdout_full"] - parent["holdout_full"]),
                      flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- rules
    print("\n=== 选型规则：留出发动机上的配对结果 ===")
    by_rule = {}
    for r in records:
        by_rule.setdefault(r["rule"], []).append(r)

    def keyed(rows):
        return {(r["seed"], r["fold"]): r for r in rows}

    ref = keyed(by_rule.get("full-only", []))
    rule_summary = {}
    for rule_name, rows in sorted(by_rule.items(),
                                  key=lambda kv: float(np.mean(
                                      [r["holdout_full"] for r in kv[1]]))):
        mean = float(np.mean([r["holdout_full"] for r in rows]))
        deltas = [r["delta"] for r in rows]
        wins = sum(1 for d in deltas if d < -1e-9)
        losses = sum(1 for d in deltas if d > 1e-9)
        entry = {
            "folds": len(rows),
            "mean_holdout_full": mean,
            "mean_delta_vs_parent": float(np.mean(deltas)),
            "beats_parent": wins, "loses_to_parent": losses,
            "selected": {},
        }
        if rule_name != "full-only" and ref:
            mine = keyed(rows)
            paired = [mine[key]["holdout_full"] - ref[key]["holdout_full"]
                      for key in sorted(mine) if key in ref]
            stat, p, n = wilcoxon(paired, [0.0] * len(paired))
            entry["paired_vs_full_only"] = {
                "mean_paired_delta": float(np.mean(paired)),
                "wins": sum(1 for d in paired if d < -1e-9),
                "losses": sum(1 for d in paired if d > 1e-9),
                "wilcoxon_stat": stat, "wilcoxon_p": p, "n": n,
                "per_fold": [{"seed": k[0], "fold": k[1],
                              "delta": d}
                             for k, d in zip(sorted(mine), paired)],
            }
        for r in rows:
            entry["selected"][r["selected"]] = entry["selected"].get(r["selected"], 0) + 1
        rule_summary[rule_name] = entry
        extra = ""
        if "paired_vs_full_only" in entry:
            p_ = entry["paired_vs_full_only"]
            extra = ("  vs full-only: %+0.4f (%d win / %d loss, Wilcoxon p=%.3f)"
                     % (p_["mean_paired_delta"], p_["wins"], p_["losses"],
                        p_["wilcoxon_p"]))
        print("  %-12s holdout %8.4f  vs parent %+7.4f  win %d/%d  %s%s"
              % (rule_name, mean, entry["mean_delta_vs_parent"],
                 wins, len(rows), extra,
                 "  sel=%s" % entry["selected"] if rule_name == "full-only" else ""))

    # ------------------------------------------------------------ recipes
    print("\n=== 超参数：与 shipped 配对（留出集 RMSE 差）===")
    recipe_rows = []
    for seed in seeds:
        folds = engine_folds(ctx, args.folds, seed=seed)
        for fold_index, held in enumerate(folds):
            held_mask = np.isin(ctx["tunit"], held)
            train_mask = ~held_mask
            base = baseline_on(ctx, held_mask)
            got = {}
            for name, spec, epochs, lr, hidden in recipes():
                model = train_on(ctx, spec, train_mask, epochs=epochs, lr=lr,
                                 batch=256, seed=seed, hidden=hidden)
                got[name] = evaluate(ctx, model, held_mask)["holdout_full_rmse"]
            recipe_rows.append({"seed": seed, "fold": fold_index,
                                "scores": got,
                                "baseline": base["holdout_full_rmse"]})
    names = [r[0] for r in recipes()]
    # Two controls.  ``shipped`` is the study's control (ep40 / lr 0.004);
    # ``prod-lr`` is the rate the production loop actually trains at, so
    # every candidate is also compared against the thing that ships.
    shipped = "shipped"
    prod = "prod-lr"
    recipe_summary = {}
    for name in names:
        deltas = [row["scores"][name] - row["scores"][shipped] for row in recipe_rows]
        deltas_prod = [row["scores"][name] - row["scores"][prod] for row in recipe_rows]
        vs_base = [row["scores"][name] - row["baseline"] for row in recipe_rows]
        base_d = [row["scores"][prod] - row["baseline"] for row in recipe_rows]
        stat, p, n = wilcoxon(deltas, [0.0] * len(deltas))
        pstat, pp, _ = wilcoxon(deltas_prod, [0.0] * len(deltas_prod))
        s_stat, s_p, _ = wilcoxon(vs_base, [0.0] * len(vs_base))
        recipe_summary[name] = {
            "folds": len(recipe_rows),
            "mean_holdout": float(np.mean([row["scores"][name] for row in recipe_rows])),
            "mean_delta_vs_shipped": float(np.mean(deltas)),
            "wins_vs_shipped": sum(1 for d in deltas if d < -1e-9),
            "losses_vs_shipped": sum(1 for d in deltas if d > 1e-9),
            "wilcoxon_vs_shipped_p": p,
            "mean_delta_vs_prod_lr": float(np.mean(deltas_prod)),
            "wins_vs_prod_lr": sum(1 for d in deltas_prod if d < -1e-9),
            "losses_vs_prod_lr": sum(1 for d in deltas_prod if d > 1e-9),
            "wilcoxon_vs_prod_lr_p": pp,
            "mean_delta_vs_baseline": float(np.mean(vs_base)),
            "beats_baseline": sum(1 for d in vs_base if d < -1e-9),
            "wilcoxon_vs_baseline_p": s_p,
            "mean_delta_prod_lr_vs_baseline": float(np.mean(base_d)),
        }
        print("  %-24s holdout %8.4f | vs shipped %+7.4f (%dW/%dL p=%.3f)"
              " | vs prod-lr %+7.4f (%dW/%dL p=%.3f) | vs base %+7.4f (win %d/%d p=%.3f)"
              % (name, recipe_summary[name]["mean_holdout"],
                 recipe_summary[name]["mean_delta_vs_shipped"],
                 recipe_summary[name]["wins_vs_shipped"],
                 recipe_summary[name]["losses_vs_shipped"], p,
                 recipe_summary[name]["mean_delta_vs_prod_lr"],
                 recipe_summary[name]["wins_vs_prod_lr"],
                 recipe_summary[name]["losses_vs_prod_lr"], pp,
                 recipe_summary[name]["mean_delta_vs_baseline"],
                 recipe_summary[name]["beats_baseline"], len(recipe_rows), s_p))

    out.write_text(json.dumps({
        "rules": {"records": records, "summary": rule_summary},
        "recipes": {"records": recipe_rows, "summary": recipe_summary},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
