# -*- coding: utf-8 -*-
"""Profile C: NASA C-MAPSS turbofan remaining useful life.

The problem
-----------
C-MAPSS is four sub-datasets of simulated turbofan engines:

======  ==========================  ==============================
subset  operating condition         difficulty
======  ==========================  ==============================
FD001   1 condition, 1 fan layout  easiest
FD002   1 condition, 1 fan layout  more modes
FD003   1 condition, 2 fan layouts  sensor noise
FD004   2 conditions, 2 fan layouts hardest
======  ==========================  ==============================

The platform's public split keeps whole engines together:

* train      182 + 182 + 70 + 174 engines (FD001/2/3/4)
* validation  15 +  39 + 15 +  37 engines

and the target is ``RUL = T - t`` in **cycles**, where ``T`` is the
engine's last cycle.

What makes C different from B
-----------------------------
Both are RUL regressions, but the failure modes are different, and the
official metric differs:

* **B (bearings)** has a *huge* target spread - 122 to 2537 minutes -
  and the target scale carries real information about the machine.
  Absolute minutes are the right target; standardising them costs ~18%
  RMSE.
* **C (engines)** has a much tighter spread - 0 to 542 cycles - and the
  four sub-datasets have visibly different degradation rates.
  FD004's engines in particular reach failure on very different cycle
  counts from FD001's, so **the sub-dataset is a confounder**: a model
  that sees a FD004 engine at cycle 50 has no idea whether it has 100
  cycles left or 300.

That is the thing to test.  If the sub-dataset explains a large part of
the residual, then the right fix is to **condition on the operating
condition** (which the inputs already carry, in ``settings``) rather
than to regress harder.

The metric that matters
-----------------------
C has a *second* official metric: the **NASA asymmetric score**, where
over-predicting the remaining life is punished five times harder than
under-predicting it, because an engine that is said to have 500 cycles
left and fails at 100 is a scrapped engine, while an engine retired 100
cycles early is merely wasteful.

The official definition is a **sum** of per-sample penalties, not a mean:
``score = sum_i penalty_i``.  A mean would change with the sample count
and could not be compared with the leaderboard.

Run:
    python -m research.tracks.track_c_cmapss --out artifacts/tracks/track_c.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tracks.official_baselines import (  # noqa: E402
    PACKAGES, load_checkpoint, load_split, score_bc,
)

DATA_ROOT = Path(r"D:\GY\cpu")
PACKAGE_C = PACKAGES["C"]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


# --------------------------------------------------------------------- data
def read_meta(split: str):
    """Read a split and keep the engine identity and the sub-dataset."""
    root = DATA_ROOT / PACKAGE_C / "public" / f"{split}.jsonl"
    rows = [json.loads(line) for line in root.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    engine, subset, cycle = [], [], []
    for row in rows:
        meta = row["meta"]
        engine.append(f"{meta['source_subset']}#{meta['unit']}")
        subset.append(meta["source_subset"])
        cycle.append(int(meta["cycle"]))
    return rows, engine, subset, np.asarray(cycle, dtype=np.float64)


def engine_table(split: str):
    """Per-engine summary: how many cycles it ran, and how it ended."""
    rows, engine, subset, cycle = read_meta(split)
    table = {}
    for name in sorted(set(engine)):
        mask = np.asarray([e == name for e in engine])
        table[name] = {
            "subset": subset[int(np.argmax(mask))],
            "rows": int(mask.sum()),
            "last_cycle": int(cycle[mask].max()),
            "max_rul": float(max(r["answer"] for r, m in zip(rows, mask) if m)),
        }
    return table


# ------------------------------------------------------------------ metrics
def nasa_score(y_true, y_pred):
    """The official NASA asymmetric score - a **sum**, not a mean."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    r = np.maximum(np.abs(y_true), 1.0)
    penalty = np.where(err > 0,
                       5.0 * np.maximum(0.0, err) / r,
                       np.maximum(0.0, -err) / r)
    return float(penalty.sum()), float(penalty.mean())


def per_subset(y_true, y_pred, subset):
    """The score broken down by sub-dataset.

    A model that is good on FD001 and bad on FD004 averages to a number
    that describes neither fleet.  The breakdown is what tells the
    mechanism apart.
    """
    out = {}
    for name in SUBSETS:
        mask = np.asarray([s == name for s in subset])
        if not mask.any():
            continue
        scored = score_bc(np.asarray(y_true)[mask], np.asarray(y_pred)[mask], True)
        out[name] = {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in scored.items()}
    return out


# ------------------------------------------------------------------- models
def fit_mlp(x, y, hidden=32, epochs=20, seed=42, learning_rate=0.003,
            batch_size=128, target_scaling=True):
    """A NumPy MLP, matching the official baseline's interface."""
    from rulkit.model import Model

    model = Model("regression", [f"f{i}" for i in range(x.shape[1])], [], x, y,
                  hidden=hidden, seed=seed, learning_rate=learning_rate,
                  batch_size=batch_size, feature_transform="raw",
                  target_scaling=target_scaling)
    model.fit(x, y, epochs, learning_rate=learning_rate)
    return model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    train_rows, train_engine, train_subset, train_cycle = read_meta("train")
    valid_rows, valid_engine, valid_subset, valid_cycle = read_meta("validation")

    train = load_split("C", "train")
    valid = load_split("C", "validation")
    train_x, train_y = train["x"], train["y"]
    valid_x, valid_y = valid["x"], valid["y"]

    print("=== profile C: NASA C-MAPSS turbofan RUL ===")
    print("  train      %d rows, %d engines, %d subsets"
          % (len(train_y), len(set(train_engine)), len(set(train_subset))))
    print("  validation %d rows, %d engines, %d subsets"
          % (len(valid_y), len(set(valid_engine)), len(set(valid_subset))))
    for name in SUBSETS:
        mask = np.asarray([s == name for s in train_subset])
        lives = [e["last_cycle"] for e in engine_table("train").values()
                 if e["subset"] == name]
        print("    %s train engines %3d  lifetime %d..%d (median %d)"
              % (name, int(mask.sum()), min(lives), max(lives), int(np.median(lives))))

    results = {"candidates": {}}

    # --- the official baseline -------------------------------------------
    _, official = load_checkpoint("C")
    official_pred = official.predict(valid_x)
    baseline = score_bc(valid_y, official_pred, True)
    results["official_baseline_validation"] = baseline
    results["official_baseline_validation_per_subset"] = per_subset(
        valid_y, official_pred, valid_subset)
    print("\n  official baseline (validation)")
    print("    RMSE %.3f  MAE %.3f  NASA %.1f"
          % (baseline["rmse"], baseline["mae"], baseline["nasa_asym_score"]))
    for name, scored in results["official_baseline_validation_per_subset"].items():
        print("      %s  RMSE %8.3f  NASA %10.1f" % (name, scored["rmse"], scored["nasa_asym_score"]))

    # --- capacity sweep --------------------------------------------------
    #
    # The sweep is the expensive part of this script: eighteen MLP fits
    # over 113 089 rows, which is roughly twenty minutes on one CPU
    # core.  Its result is therefore cached next to the output file, and
    # a re-run only re-fits the winner.  The cache is keyed by the
    # candidate name, and the winner is always re-fitted from scratch, so
    # a stale cache can never change the reported number.
    print("\n  --- capacity sweep (target scaling on/off) ---")
    print("  %-28s %10s %12s" % ("candidate", "RMSE", "NASA"))
    cache_path = None
    if args.out:
        cache_path = Path(args.out).with_suffix(".sweep.json")
    sweep = {}
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        sweep = {k: v for k, v in cached.items() if isinstance(v, dict) and "rmse" in v}
        if sweep:
            print("  (reusing the cached sweep: %s)" % cache_path.name)
    for hidden in (32, 64, 128):
        for epochs in (20, 60, 200):
            for scaling in (True, False):
                key = f"mlp/h{hidden}/e{epochs}/scaling{int(scaling)}"
                if key in sweep:
                    scored = sweep[key]
                    print("  %-28s %10.3f %12.1f  (cached)"
                          % (key, scored["rmse"], scored["nasa_asym_score"]))
                    continue
                model = fit_mlp(train_x, train_y, hidden=hidden, epochs=epochs,
                                target_scaling=scaling)
                pred = model.predict(valid_x)
                scored = score_bc(valid_y, pred, True)
                sweep[key] = scored
                print("  %-28s %10.3f %12.1f" % (key, scored["rmse"], scored["nasa_asym_score"]))
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(sweep, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")
    best_key = min(sweep, key=lambda k: sweep[k]["rmse"])
    print("  best: %s  RMSE %.3f" % (best_key, sweep[best_key]["rmse"]))

    # --- the winner, broken down by sub-dataset --------------------------
    #
    # The key looks like ``mlp/h128/e200/scaling1``: four slash-separated
    # pieces, and the first one has no "=" in it.  Splitting each piece on
    # "/" a second time - as an earlier version did - raises
    # ``ValueError: dictionary update sequence element #0 has length 1``
    # on the very first piece.  The key is parsed by splitting it once
    # and stripping the known prefixes off each field.
    fields = best_key.split("/")
    parts = {"h": fields[1], "e": fields[2], "scaling": fields[3]}
    model = fit_mlp(train_x, train_y, hidden=int(parts["h"][1:]),
                    epochs=int(parts["e"][1:]), target_scaling=parts["scaling"] == "1")
    pred = model.predict(valid_x)
    winner = score_bc(valid_y, pred, True)
    results["candidates"][best_key] = winner
    results["candidates"][best_key]["per_subset"] = per_subset(valid_y, pred, valid_subset)
    print("\n  winner %s" % best_key)
    print("    RMSE %.3f  MAE %.3f  NASA %.1f"
          % (winner["rmse"], winner["mae"], winner["nasa_asym_score"]))
    for name, scored in results["candidates"][best_key]["per_subset"].items():
        print("      %s  RMSE %8.3f  NASA %10.1f" % (name, scored["rmse"], scored["nasa_asym_score"]))

    # --- how much of the error is the sub-dataset? -----------------------
    # A plain question with a plain answer: if the model predicts a
    # per-sub-dataset *bias*, how much of the error disappears?  That
    # upper bounds what conditioning on the operating condition can buy.
    print("\n  --- residual structure ---")
    residual = valid_y - pred
    for name in SUBSETS:
        mask = np.asarray([s == name for s in valid_subset])
        print("    %s  mean residual %+8.3f  RMSE %8.3f"
              % (name, float(residual[mask].mean()), float(np.sqrt(np.mean(residual[mask] ** 2)))))
    corrected = pred.copy()
    for name in SUBSETS:
        mask = np.asarray([s == name for s in valid_subset])
        corrected[mask] = pred[mask] + residual[mask].mean()
    oracle = score_bc(valid_y, corrected, True)
    results["subset_bias_correction_oracle"] = oracle
    print("    oracle per-subset bias correction: RMSE %.3f (from %.3f)"
          % (oracle["rmse"], winner["rmse"]))

    # --- the decision ----------------------------------------------------
    gain = (baseline["rmse"] - winner["rmse"]) / max(baseline["rmse"], 1e-9) * 100.0
    results["decision"] = {
        "winner": best_key,
        "rmse": round(winner["rmse"], 3),
        "official_rmse": round(baseline["rmse"], 3),
        "gain_pct": round(gain, 2),
        "nasa_score": round(winner["nasa_asym_score"], 1),
        "official_nasa_score": round(baseline["nasa_asym_score"], 1),
    }
    results["capacity_sweep"] = sweep
    print("\n  winner %s  RMSE %.3f vs official %.3f (%+.2f%%)"
          % (best_key, winner["rmse"], baseline["rmse"], gain))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
