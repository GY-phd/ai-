"""Fusion probe 3: isolate WHY sklearn-ReLU+slog beats AI2-tanh+slog.

Probe 1 result (official holdout bearings, RMSE in minutes):
    raw   + sklearn MLP(256) = 521.07
    slog  + sklearn MLP(256) = 404.45   <-- best
    eng34 + sklearn MLP(256) = 771.21
    slog  + AI2 tanh MLP(96) = 535.09   (official checkpoint format)

Three candidate explanations for the 404 vs 535 gap:
    H1  activation     - ReLU beats tanh for this regression
    H2  target scaling - sklearn path does NOT scale the target; AI2 does
    H3  capacity       - sklearn hidden 256 vs AI2 hidden 96

This script holds the model class fixed (sklearn MLPRegressor) and flips one
factor at a time, so the attribution is unambiguous. The winner then becomes
the specification for the merged implementation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

AI2 = Path(r"D:/VSCode project/deepseek/AI2-github")
DATA = Path(r"D:/GY/deepseek/cpu-b-rul-20261110-v1/public")

if str(AI2) not in sys.path:
    sys.path.insert(0, str(AI2))


def _load_jsonl(path: Path):
    """Read a JSONL file into a list of dicts."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build():
    """Return raw X/y for train and validation."""
    from rulkit.features import feature_matrix

    out = {}
    for name in ("train", "validation"):
        rows = _load_jsonl(DATA / f"{name}.jsonl")
        X, paths = feature_matrix([r["inputs"] for r in rows])
        y = np.asarray([float(r["answer"]) for r in rows], dtype=np.float64)
        out[name] = (X, y)
    return out


def _rmse(y, p):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _mae(y, p):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _pipeline(hidden, alpha=1e-4, activation="relu", max_iter=1200, lr=3e-3, seed=42):
    """Input-standardised MLP pipeline (target left in original minutes)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
            random_state=seed, learning_rate_init=lr,
            early_stopping=False, solver="adam",
            activation=activation,
        )),
    ])


def main() -> int:
    """Run the factor-isolation probe."""
    from rulkit.features import transform

    data = _build()
    Xtr, ytr = data["train"]
    Xva, yva = data["validation"]
    print(f"train X={Xtr.shape}  valid X={Xva.shape}\n", flush=True)

    results = {}

    def run(label, Xtr_t, Xva_t, ytr_t, yva_t, pipe):
        """Fit one configuration and record its score."""
        t0 = time.perf_counter()
        pipe.fit(Xtr_t, ytr_t)
        pred = np.clip(pipe.predict(Xva_t), 0, None)
        r = _rmse(yva_t, pred)
        results[label] = r
        print(
            f"  {label:42s} RMSE={r:8.2f}  MAE={_mae(yva_t, pred):8.2f}"
            f"  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    Ttr = transform(Xtr, "slog")
    Tva = transform(Xva, "slog")

    print("=== H1: activation (relu vs tanh), h256, slog, raw target ===", flush=True)
    run("H1 relu  h256  slog  raw-target", Ttr, Tva, ytr, yva,
        _pipeline((256,), activation="relu"))
    run("H1 tanh  h256  slog  raw-target", Ttr, Tva, ytr, yva,
        _pipeline((256,), activation="tanh"))
    print()

    print("=== H2: target scaling (standardised vs raw minutes) ===", flush=True)
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler as _SS

    ttr = TransformedTargetRegressor(
        regressor=_pipeline((256,), activation="relu"), transformer=_SS()
    )
    run("H2 relu  h256  slog  scaled-target", Ttr, Tva, ytr, yva, ttr)
    print()

    print("=== H3: capacity (slog, relu, raw target) ===", flush=True)
    for hidden in (64, 96, 128, 256, 384):
        run(f"H3 relu  h{hidden:<3d} slog  raw-target",
            Ttr, Tva, ytr, yva, _pipeline((hidden,)))
    print()

    print("=== H3b: two-layer (slog, relu, raw target) ===", flush=True)
    for hidden in ((256, 128), (384, 192), (128, 64)):
        run(f"H3b relu  {str(hidden):12s} slog raw-target",
            Ttr, Tva, ytr, yva, _pipeline(hidden))
    print()

    print("=== Ranking ===", flush=True)
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:42s} {v:9.2f}", flush=True)
    best = min(results.items(), key=lambda kv: kv[1])
    print(f"\n--> best: {best[0]} RMSE={best[1]:.2f}", flush=True)
    print(
        f"    vs official static baseline 681.16"
        f"  ->  {(681.16 - best[1]) / 681.16 * 100:.2f}% better"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
