# -*- coding: utf-8 -*-
"""Can the shipped official checkpoint serve as the calibration ruler for C?

The question
------------
Profile C's hidden score (56.253) was worse than the shipped static
baseline, and the rules compare against that baseline.  The public split
cannot see the hidden pack, so the question is what *offline* instrument
can be trusted to predict the hidden outcome.

Two candidate instruments:

``public split``
    What the runtime optimises today.  The 2026-11-23 run showed it can
    be wrong: the deployed candidate beat the baseline on the public
    split (51.111 vs 52.869) and lost on the hidden split.

``leave-one-engine-out``
    Train on the public training engines, evaluate on engines the model
    has never seen.  The hidden pack is a different dataset, so this is
    the closest available stand-in for "an engine the system has never
    seen".

The calibration ruler
---------------------
The shipped ``checkpoint.npz`` *is* the thing the rules compare against.
Scoring it under each protocol tells us which protocol reproduces the
baseline's behaviour.  A protocol under which the baseline looks strong
is a protocol that is measuring the right thing; one under which the
baseline looks weak is measuring something the hidden split does not
reward.

The decisive table
------------------
For every candidate recipe, the study reports the candidate's score and
the baseline's score **under the same protocol**, and asks the question
the rules ask: *did the candidate end up below the baseline?*

A protocol is only useful if the answer it gives is stable - i.e. if it
does not flip between "beats the baseline" and "loses to the baseline"
depending on which recipe happened to be tried.

Run::

    python -m research.tracks.c_calibrate --folds 5 --epochs 30 --seeds 42,7
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


def recipes():
    """The candidate recipes that were really on the table."""
    return [
        ("control", dict(w="none")),
        ("surv150", dict(w="surv", H=150, scale=75, floor=0.05)),
        ("surv150p05", dict(w="surv", H=150, scale=75, floor=0.05, power=0.5)),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_calibrate.json"))
    args = parser.parse_args(argv)

    ctx = load(args.package or None)
    folds = engine_folds(ctx, args.folds)
    seeds = [int(s) for s in args.seeds.split(",")]
    print("public train engines %d -> %d folds; seeds %s"
          % (len(ctx["t_engines"]), args.folds, seeds), flush=True)

    rows = []
    for seed in seeds:
        for fold_index, fold in enumerate(folds):
            held = np.isin(ctx["tunit"], fold)
            train_mask = ~held

            base = baseline_on(ctx, held)
            base_public = baseline_on(ctx, np.ones(len(ctx["train"]["y"]), dtype=bool))
            # The baseline's public-split score is the same in every fold;
            # it is recorded once for the cross-protocol table.
            base_public_full = base_public["holdout_full_rmse"]

            for name, spec in recipes():
                started = time.time()
                model = train_on(ctx, spec, train_mask, epochs=args.epochs,
                                 lr=0.004, batch=256, seed=seed)
                held_scores = evaluate(ctx, model, held)

                # Public-split score of the same model.
                vy = ctx["valid"]["y"]
                vp = model.predict(ctx["valid"]["x"])
                public_full = rmse(vy, vp)
                le = vy <= 200
                public_late = rmse(vy[le], vp[le])

                rows.append({
                    "seed": seed, "fold": fold_index, "recipe": name, "spec": spec,
                    "seconds": round(time.time() - started, 1),
                    "holdout_full": held_scores["holdout_full_rmse"],
                    "holdout_late200": held_scores["holdout_late200_rmse"],
                    "holdout_mae": held_scores["holdout_mae"],
                    "holdout_nasa_mean": held_scores["holdout_nasa_mean"],
                    "baseline_holdout_full": base["holdout_full_rmse"],
                    "baseline_holdout_late200": base["holdout_late200_rmse"],
                    "public_full": public_full,
                    "public_late200": public_late,
                    "baseline_public_full": base_public_full,
                })
                print("  s%d fold%d %-11s holdout %8.4f (base %8.4f)  public %8.4f (base %8.4f)  %.0fs"
                      % (seed, fold_index, name, held_scores["holdout_full_rmse"],
                         base["holdout_full_rmse"], public_full, base_public_full,
                         time.time() - started), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
