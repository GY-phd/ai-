"""Fusion probe: does the AI2 `eng34` feature transform + our MLP stack up?

Two candidate sources are combined here:

  * AI2-github `rulkit/features.py` — the dimension-preserving `eng34`
    reparameterisation (log magnitudes + dimensionless ratios). It keeps the
    official 34-column schema intact, so the official checkpoint loader can
    still read the model.
  * ai+ `evo` package — the sklearn-based MLP candidate model, the
    input-standardisation pipeline, and the holdout-bearing evaluation
    protocol.

The question this script answers: does `eng34` + MLP beat either alone?

Evaluation protocol (identical to both upstream repos):
    train  = train.jsonl      (9 bearings)
    eval   = validation.jsonl (3 fully isolated bearings, n=1679)
    metric = RMSE in minutes (official primary metric)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

AI2 = Path(r"D:/VSCode project/deepseek/AI2-github")
DATA = Path(r"D:/GY/deepseek/cpu-b-rul-20261110-v1/public")

# Make AI2's `rulkit` importable at module load time (before any function
# body runs) — otherwise `from rulkit.features import ...` inside main()
# fails even though _build() would have inserted the path later.
if str(AI2) not in sys.path:
    sys.path.insert(0, str(AI2))


def _load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build():
    """Return raw X/y for train and validation plus bearing labels."""
    if str(AI2) not in sys.path:
        sys.path.insert(0, str(AI2))
    from rulkit.features import feature_matrix

    out = {}
    for name in ("train", "validation"):
        rows = _load_jsonl(DATA / f"{name}.jsonl")
        X, paths = feature_matrix([r["inputs"] for r in rows])
        y = np.asarray([float(r["answer"]) for r in rows], dtype=np.float64)
        bearings = [r.get("meta", {}).get("bearing", "unknown") for r in rows]
        out[name] = (X, y, bearings, paths)
    return out


def _rmse(y, p):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _mae(y, p):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _mlp(hidden, alpha=1e-4, max_iter=1200, lr=3e-3, seed=42):
    """Input-standardised MLP pipeline (no target scaling — it hurt RUL)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
            random_state=seed, learning_rate_init=lr,
            early_stopping=False, solver="adam",
        )),
    ])


def main() -> int:
    """Run the fusion probe."""
    from rulkit.features import transform

    data = _build()
    Xtr, ytr, btr, _ = data["train"]
    Xva, yva, bva, _ = data["validation"]
    print(f"train X={Xtr.shape}  valid X={Xva.shape}", flush=True)
    print(f"valid bearings: {sorted(set(bva))}", flush=True)
    print()

    results = {}

    for mode in ("raw", "slog", "eng34"):
        Ttr = transform(Xtr, mode)
        Tva = transform(Xva, mode)
        for label, hidden, alpha in (
            ("mlp(256)", (256,), 1e-4),
            ("mlp(128)", (128,), 1e-3),
            ("mlp(512)", (512,), 1e-3),
        ):
            t0 = time.perf_counter()
            pipe = _mlp(hidden, alpha)
            pipe.fit(Ttr, ytr)
            p = np.clip(pipe.predict(Tva), 0, None)
            r = _rmse(yva, p)
            m = _mae(yva, p)
            key = f"{mode}+{label}"
            results[key] = r
            print(
                f"  {key:26s} RMSE={r:8.2f}  MAE={m:8.2f}  ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        print()

    print("=== Ranking ===", flush=True)
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:26s} {v:9.2f}", flush=True)

    best = min(results.items(), key=lambda kv: kv[1])
    print(f"\n--> best: {best[0]} RMSE={best[1]:.2f}", flush=True)

    # Per-bearing breakdown for the winner.
    mode, label = best[0].split("+")
    hidden, alpha = {
        "mlp(256)": ((256,), 1e-4),
        "mlp(128)": ((128,), 1e-3),
        "mlp(512)": ((512,), 1e-3),
    }[label]
    pipe = _mlp(hidden, alpha)
    pipe.fit(transform(Xtr, mode), ytr)
    pred = np.clip(pipe.predict(transform(Xva, mode)), 0, None)
    print("\n=== Per-bearing (winner) ===", flush=True)
    for b in sorted(set(bva)):
        mask = np.asarray([x == b for x in bva])
        print(
            f"  {b:16s} n={int(mask.sum()):5d}  RMSE={_rmse(yva[mask], pred[mask]):8.2f}"
            f"  MAE={_mae(yva[mask], pred[mask]):8.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
