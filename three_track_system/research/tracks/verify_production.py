# -*- coding: utf-8 -*-
"""Cross-check the production two-stage model against the research protocol.

The production model in ``rulkit/rul.py`` and the research prototype in
``research/two_stage.py`` were written separately.  A number that only
exists in the research script is not a result - it is an anecdote.  This
script fits the **production** model under the **research** protocol and
asserts that the two agree, so the number that goes into the report is
the number the shipped code produces.

The check runs leave-one-bearing-out over the nine training bearings and
compares:

* the production ``LifeFractionModel`` with ``normalisation="slope"``;
* the research prototype's ``safe_lifetime`` + slope features.

They must agree to within numerical noise.  If they do not, one of them
has drifted from the other and the reported figure is meaningless.

Run:
    python -m research.tracks.verify_production --out artifacts/tracks/production_check.json
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

from research.protocol import bearing_groups, load_split, subset  # noqa: E402
from research.selfnorm_study import self_normalise  # noqa: E402
from research.two_stage import life_fraction  # noqa: E402
from rulkit.model import Model  # noqa: E402
from rulkit.rul import LifeFractionModel, safe_lifetime  # noqa: E402


CONFIG = {
    "hidden": 96,
    "epochs": 400,
    "learning_rate": 0.003,
    "weight_decay": 0.01,
    "batch_size": 128,
    "feature_transform": "eng34",
    "activation": "tanh",
    "target_scaling": False,
    "seed": 42,
    "floor": 0.02,
    "cap": 3000.0,
    "rate_limit": 0.25,
}


def research_reference(train_rows, eval_rows, config):
    """The research prototype's prediction path, verbatim."""
    matrix_train = self_normalise(train_rows, "slope", 5)
    matrix_eval = self_normalise(eval_rows, "slope", 5)
    y_train = life_fraction(train_rows)
    model = Model("regression", train_rows["paths"], [], matrix_train, y_train,
                  hidden=config["hidden"], seed=config["seed"],
                  learning_rate=config["learning_rate"], batch_size=config["batch_size"],
                  feature_transform=config["feature_transform"],
                  activation=config["activation"], target_scaling=False)
    model.fit(matrix_train, y_train, config["epochs"],
              learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])

    fraction = np.clip(model.predict(matrix_eval), 0.0, 1.0)
    scale = np.zeros(len(eval_rows["y"]))
    for _, indices in sorted(bearing_groups(eval_rows).items()):
        order = indices[np.argsort(eval_rows["snapshot"][indices])]
        elapsed = np.arange(1, len(order) + 1, dtype=float)
        scale[order] = safe_lifetime(fraction[order], elapsed,
                                     floor=config["floor"], cap=config["cap"],
                                     rate_limit=config["rate_limit"])
    return np.clip(fraction * scale, 0.0, config["cap"])


def production_model(train_rows, eval_rows, config):
    """The shipped ``LifeFractionModel``, used the way the harness uses it."""
    groups_train = _groups(train_rows)
    groups_eval = _groups(eval_rows)
    elapsed_train = _elapsed(train_rows, groups_train)
    elapsed_eval = _elapsed(eval_rows, groups_eval)

    model = LifeFractionModel(
        floor=config["floor"], cap=config["cap"], rate_limit=config["rate_limit"],
        max_rul_minutes=config["cap"], normalisation="slope", window=5,
        smooth_window=1)
    fraction = life_fraction(train_rows)
    model.fit(train_rows["x"], fraction, train_rows["paths"], groups_train,
              hidden=config["hidden"], seed=config["seed"],
              learning_rate=config["learning_rate"], batch_size=config["batch_size"],
              feature_transform=config["feature_transform"],
              activation=config["activation"], epochs=config["epochs"],
              weight_decay=config["weight_decay"], elapsed=elapsed_train)
    return model.predict(eval_rows["x"], groups_eval, elapsed_eval)


def _groups(rows):
    names = sorted(bearing_groups(rows))
    lookup = {}
    for name in names:
        for index in bearing_groups(rows)[name]:
            lookup[index] = name
    return np.asarray([lookup[i] for i in range(len(rows["y"]))], dtype=object)


def _elapsed(rows, groups):
    """The elapsed-snapshot count of each row inside its own bearing."""
    out = np.zeros(len(groups), dtype=np.float64)
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        order = indices[np.argsort(rows["snapshot"][indices])]
        out[order] = np.arange(1, len(order) + 1, dtype=float)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=r"D:\GY\cpu\cpu-b-rul-20261110-v1")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = load_split(Path(args.data), "train")
    groups = bearing_groups(rows)

    research_rmse, production_rmse, deltas = [], [], []
    print("=== production vs research, leave-one-bearing-out ===")
    print("  %-18s %12s %12s %12s" % ("held-out bearing", "research", "production", "|delta|"))
    for held_out in sorted(groups):
        eval_indices = groups[held_out]
        train_indices = np.asarray([i for name, idx in groups.items() if name != held_out
                                    for i in idx])
        train_rows, eval_rows = subset(rows, train_indices), subset(rows, eval_indices)

        reference = research_reference(train_rows, eval_rows, CONFIG)
        production = production_model(train_rows, eval_rows, CONFIG)

        r = float(np.sqrt(np.mean((reference - eval_rows["y"]) ** 2)))
        p = float(np.sqrt(np.mean((production - eval_rows["y"]) ** 2)))
        research_rmse.append(r)
        production_rmse.append(p)
        deltas.append(abs(r - p))
        print("  %-18s %12.3f %12.3f %12.3f" % (held_out, r, p, abs(r - p)))

    research_rmse = np.asarray(research_rmse)
    production_rmse = np.asarray(production_rmse)
    max_delta = float(np.max(deltas))
    agree = max_delta <= 1.0  # 1 minute out of a ~380 minute score

    print("\n=== summary ===")
    print("  research   mean %8.2f  median %8.2f  worst %8.2f"
          % (research_rmse.mean(), np.median(research_rmse), research_rmse.max()))
    print("  production mean %8.2f  median %8.2f  worst %8.2f"
          % (production_rmse.mean(), np.median(production_rmse), production_rmse.max()))
    print("  max |delta| %.4f min  ->  %s" % (max_delta, "AGREE" if agree else "MISMATCH"))

    result = {
        "research_rmse_mean": round(float(research_rmse.mean()), 3),
        "research_rmse_median": round(float(np.median(research_rmse)), 3),
        "research_rmse_worst": round(float(research_rmse.max()), 3),
        "production_rmse_mean": round(float(production_rmse.mean()), 3),
        "production_rmse_median": round(float(np.median(production_rmse)), 3),
        "production_rmse_worst": round(float(production_rmse.max()), 3),
        "max_abs_delta_minutes": round(max_delta, 4),
        "agree": bool(agree),
        "folds": [
            {"bearing": b,
             "research_rmse": round(r, 3),
             "production_rmse": round(p, 3)}
            for b, r, p in zip(sorted(groups), research_rmse, production_rmse)
        ],
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
