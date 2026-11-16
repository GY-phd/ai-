# -*- coding: utf-8 -*-
"""Where is the degradation signal in the 34 official features?

The leave-one-bearing-out result (639 RMSE, barely better than the official
baseline's 681) says the current feature set does not identify degradation
well enough to predict remaining life across bearings.  Before trying new
models, it is worth asking what the 34 features actually contain.

This script measures, per bearing, how each feature correlates with the
degradation proxy (the label-free index defined in ``target_study``).  A
feature that correlates strongly *within* every bearing but has a different
relationship between bearings is useful for "how degraded is this now" but
useless for "how long is left".  A feature whose correlation is weak
everywhere is simply noise for this task.

The output is the evidence needed to justify the feature work that follows:
which channels carry signal, and whether the signal is consistent across
bearings (which is the condition for any model to generalise).

Run:
    python -m research.feature_study --out artifacts/feature_study.json
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

from research.protocol import bearing_groups, load_split  # noqa: E402
from research.target_study import degradation_proxy  # noqa: E402


def pearson(a, b) -> float:
    """Pearson correlation, 0 when either side is constant."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    """Rank correlation - the degradation relation need not be linear."""
    def rank(v):
        order = np.argsort(v)
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # average the ranks of ties
        unique, inverse = np.unique(v, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]

    return pearson(rank(a), rank(b))


def main(argv=None) -> int:
    """Report per-feature correlation with degradation, across bearings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)
    degradation = degradation_proxy(rows)
    names = _feature_names(rows["paths"])

    print("=== per-feature correlation with the degradation index ===")
    print("  (each bearing separately; |rho| consistent across bearings = transferable signal)\n")
    print("  %-28s %8s %8s %8s %8s" % ("feature", "mean|r|", "min|r|", "max|r|", "sign"))
    records = []
    for index, name in enumerate(names):
        rhos = []
        for _, indices in sorted(groups.items()):
            rhos.append(spearman(rows["x"][indices, index], degradation[indices]))
        rhos = np.asarray(rhos)
        signs = np.sign(rhos[np.abs(rhos) > 0.1])
        consistent = bool(len(set(signs.tolist())) <= 1) if len(signs) else False
        records.append({
            "feature": name,
            "rho_mean": round(float(rhos.mean()), 4),
            "rho_abs_mean": round(float(np.abs(rhos).mean()), 4),
            "rho_min": round(float(rhos.min()), 4),
            "rho_max": round(float(rhos.max()), 4),
            "consistent_sign": consistent,
        })
        print("  %-28s %8.3f %8.3f %8.3f %8s" % (
            name, np.abs(rhos).mean(), np.abs(rhos).min(), np.abs(rhos).max(),
            "yes" if consistent else "NO"))

    records.sort(key=lambda r: -r["rho_abs_mean"])
    print("\n=== strongest transferable features ===")
    for row in records[:10]:
        print("  %-28s |rho| %.3f  range [%.3f, %.3f]  consistent=%s" % (
            row["feature"], row["rho_abs_mean"], row["rho_min"], row["rho_max"], row["consistent_sign"]))

    consistent = [r for r in records if r["consistent_sign"] and r["rho_abs_mean"] > 0.3]
    print("\n  %d features have consistent sign and |rho| > 0.30 across all 9 bearings" % len(consistent))
    for row in consistent:
        print("    %-28s |rho| %.3f" % (row["feature"], row["rho_abs_mean"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"features": records, "n_consistent": len(consistent)},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\nwrote %s" % out)
    return 0


def _feature_names(paths) -> list:
    """Turn the flattened paths into readable feature names."""
    names = []
    for path in paths:
        try:
            parts = json.loads(path)
            keys = [p[1] for p in parts if p[0] == "key"]
            names.append(".".join(keys))
        except Exception:  # noqa: BLE001 - the path format is fixed, but be safe
            names.append(path)
    return names


if __name__ == "__main__":
    sys.exit(main())
