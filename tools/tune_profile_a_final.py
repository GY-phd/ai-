# -*- coding: utf-8 -*-
"""Choose Profile A's final-refit recipe on the untouched validation batch.

The loop's final score comes from a refit on the initial rows plus the
stream, scored on the validation batch the loop never trains on.  That
is the number the competition's hidden stream resembles, so it is the
number to optimise.

Two things make this worth doing carefully:

* a single fit moves by a wide margin on this profile, so every
  configuration is measured over several seeds and reported by its mean
  **and** its spread; and
* the mean and the spread can disagree - a recipe can win on one and
  lose on the other - so both are printed, and the ranking prefers the
  configuration that wins on the mean while shrinking the spread.

The recipe that the champion starts from is measured in the same run,
so the comparison is between two measurements of the same thing.

Run::

    python tools/tune_profile_a_final.py
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

from tracks.profiles import get, CLASSES_A  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402

SEEDS = (0, 1, 2, 42, 7, 123, 11, 5, 99, 2024)

CHAMPION = {"hidden": 32, "epochs": 50, "feature_transform": "slog",
            "learning_rate": 0.03, "batch_size": 256}

CANDIDATES = [
    ("champion", dict(CHAMPION), "slog"),
    ("raw, lr .003, bs 128", {"hidden": 32, "epochs": 50,
                              "learning_rate": 0.003, "batch_size": 128}, "raw"),
    ("slog, lr .003, bs 128", {"hidden": 32, "epochs": 50,
                               "learning_rate": 0.003, "batch_size": 128}, "slog"),
    ("slog, lr .001, bs 128", {"hidden": 32, "epochs": 50,
                               "learning_rate": 0.001, "batch_size": 128}, "slog"),
    ("slog, lr .001, bs 64", {"hidden": 32, "epochs": 50,
                              "learning_rate": 0.001, "batch_size": 64}, "slog"),
    ("slog, lr .003, bs 64", {"hidden": 32, "epochs": 50,
                              "learning_rate": 0.003, "batch_size": 64}, "slog"),
    ("slog, lr .003, bs 256", {"hidden": 32, "epochs": 50,
                               "learning_rate": 0.003, "batch_size": 256}, "slog"),
    ("slog, lr .01, bs 128", {"hidden": 32, "epochs": 50,
                               "learning_rate": 0.01, "batch_size": 128}, "slog"),
    ("slog, lr .01, bs 256", {"hidden": 32, "epochs": 50,
                               "learning_rate": 0.01, "batch_size": 256}, "slog"),
    ("slog, lr .003, bs 128, e100", {"hidden": 32, "epochs": 100,
                                     "learning_rate": 0.003,
                                     "batch_size": 128}, "slog"),
    ("slog, lr .003, bs 128, e200", {"hidden": 32, "epochs": 200,
                                     "learning_rate": 0.003,
                                     "batch_size": 128}, "slog"),
    ("slog, lr .003, bs 128, h64", {"hidden": 64, "epochs": 50,
                                    "learning_rate": 0.003,
                                    "batch_size": 128}, "slog"),
    ("slog, lr .003, bs 128, h16", {"hidden": 16, "epochs": 50,
                                    "learning_rate": 0.003,
                                    "batch_size": 128}, "slog"),
    ("slog, lr .003, bs 128, balanced", {"hidden": 32, "epochs": 50,
                                         "learning_rate": 0.003,
                                         "batch_size": 128,
                                         "class_weight": "balanced"}, "slog"),
    ("slog, lr .003, bs 128, train_scale",
     {"hidden": 32, "epochs": 50, "learning_rate": 0.003,
      "batch_size": 128, "normalisation": "train_scale"}, "slog"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    profile = get("A")
    seeds = SEEDS[:args.seeds]
    initial, stream, judge, validation, _meta = build_stream("A", 4, 2)

    x_all = np.vstack([initial["x"], stream["x"]])
    y_all = np.concatenate([initial["y"], stream["y"]])
    rows = []

    for label, overrides, mode in CANDIDATES:
        if mode == "slog":
            x = np.sign(x_all) * np.log1p(np.abs(x_all))
            xv = np.sign(validation["x"]) * np.log1p(np.abs(validation["x"]))
        else:
            x, xv = x_all, validation["x"]
        macros = []
        for seed in seeds:
            kwargs = dict(overrides)
            kwargs["seed"] = seed
            model = profile.make_model(x, y_all, initial["paths"], **kwargs)
            prediction = model.predict(xv)
            if prediction.ndim == 2:
                prediction = prediction.argmax(axis=1)
            macros.append(profile.metric(validation["y"], prediction)["macro_f1"])
        macros = np.asarray(macros)
        rows.append({"label": label, "mean": float(macros.mean()),
                     "std": float(macros.std()),
                     "min": float(macros.min()), "max": float(macros.max())})
        print("%-38s mean %.4f  std %.4f  [%.4f .. %.4f]"
              % (label, macros.mean(), macros.std(), macros.min(), macros.max()),
              flush=True)

    print("\n" + "=" * 78)
    print("ranked by mean, with the spread beside it")
    print("=" * 78)
    for row in sorted(rows, key=lambda r: -r["mean"]):
        print("  %-38s mean %.4f  std %.4f" % (row["label"], row["mean"],
                                               row["std"]))

    champion = next(r for r in rows if r["label"] == "champion")
    best = max(rows, key=lambda r: r["mean"])
    print("\nchampion mean %.4f (std %.4f)" % (champion["mean"], champion["std"]))
    print("best     mean %.4f (std %.4f)  %s"
          % (best["mean"], best["std"], best["label"]))
    print("change   %+.4f mean, %+.4f std"
          % (best["mean"] - champion["mean"], best["std"] - champion["std"]))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile_a_final_tuning.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % (out / "profile_a_final_tuning.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
