"""One C weighting configuration, scored on the hidden-protocol proxies.

This is the shared worker for the search workflow.  It trains one
checkpoint from a parent with a configurable weight prior, schedule and
target transform, then reports:

* ``full_rmse`` - every row of the public validation split;
* ``late_rmse`` - rows with true RUL <= 200 (the profile's selection view);
* ``uniform_protocol`` - one uniformly chosen truncation horizon per engine;
* ``cmapss_proxy`` - the empirical public C-MAPSS test RUL distribution.

The weight construction is deliberately the same as
``contest_baselines.strategies._bounded_prior_weights`` for the uniform
shape, so a research result transfers to the submission path.

Run::

    python -m research.tracks.c_weight_sweep --shape uniform --horizon 150 \
        --floor 0.15 --power 0.5 --lr 0.002 --epochs 30 \
        --parent runs/harness-offline/checkpoints/C-round-1.npz

The search that produced the current production weights is recorded in
``artifacts/tracks/c_weight_search.json`` (60+ configurations).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.tracks import c_weight as cw  # noqa: E402
from contest_baselines.model import Model  # noqa: E402
from contest_baselines.strategies import _bounded_prior_weights  # noqa: E402


def target_weights(y, *, shape, horizon, tau=80.0, power=0.5, floor=0.20, bandwidth=8.0):
    """Same construction as the production helper, with a shape choice."""
    values = np.asarray(y, dtype=np.float64)
    top = int(max(np.ceil(float(values.max())), np.ceil(float(horizon))))
    grid = np.arange(0, top + 1, dtype=float)
    counts = np.bincount(np.clip(np.round(values).astype(int), 0, top),
                         minlength=top + 1).astype(np.float64)
    kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / float(bandwidth)) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    density = kernel @ counts
    density /= density.sum()
    if shape == "uniform":
        target = (grid <= float(horizon)).astype(np.float64)
    elif shape == "tri":
        target = np.where(grid <= float(horizon), float(horizon) - grid + 1.0, 0.0)
    elif shape == "exp":
        target = np.where(grid <= float(horizon), np.exp(-grid / float(tau)), 0.0)
    else:
        raise ValueError(shape)
    target /= target.sum()
    ratio = np.clip(target / (density + 1e-12), float(floor), 20.0)
    weights = ratio[np.clip(np.round(values).astype(int), 0, top)] ** float(power)
    return weights / weights.mean()


def uniform_protocol(ctx, pred, horizon=200.0):
    yv = ctx["valid"]["y"]
    import json as _json
    rows = [_json.loads(line) for line in
            (ctx["pkg"] / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    vunit = np.array([f"{r['meta']['source_subset']}#{r['meta']['unit']}" for r in rows])
    per = []
    for unit in sorted(set(vunit.tolist())):
        idx = np.nonzero(vunit == unit)[0]
        by_rul = {int(round(yv[k])): k for k in idx}
        top = min(int(horizon), max(by_rul))
        errors = [float(pred[by_rul[h]] - yv[by_rul[h]]) ** 2
                  for h in range(top + 1) if h in by_rul]
        if errors:
            per.append(float(np.mean(errors)))
    return float(np.sqrt(np.mean(per)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="")
    parser.add_argument("--shape", default="uniform", choices=("uniform", "tri", "exp"))
    parser.add_argument("--horizon", type=float, default=200.0)
    parser.add_argument("--tau", type=float, default=80.0)
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--floor", type=float, default=0.20)
    parser.add_argument("--bandwidth", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="unweighted epochs before the weighted phase")
    parser.add_argument("--cap", type=float, default=0.0,
                        help="cap the training target at this RUL (0 = no cap)")
    parser.add_argument("--subset-prior", action="store_true",
                        help="fit the target density separately per FD subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    ctx = cw.load()
    parent = args.parent or str(ctx["pkg"] / "checkpoint.npz")
    model = Model.load(parent)
    model.meta["learning_rate"] = args.lr
    model.meta["batch_size"] = args.batch
    model.meta["seed"] = args.seed
    model.validate()

    y = ctx["train"]["y"]
    target = y if args.cap <= 0 else np.minimum(y, float(args.cap))
    if args.subset_prior:
        tsub = ctx["tsub"]
        weights = np.empty(len(y), dtype=np.float64)
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            mask = tsub == subset
            if mask.any():
                weights[mask] = target_weights(
                    y[mask], shape=args.shape, horizon=args.horizon, tau=args.tau,
                    power=args.power, floor=args.floor, bandwidth=args.bandwidth)
        weights /= weights.mean()
    else:
        weights = target_weights(
            y, shape=args.shape, horizon=args.horizon, tau=args.tau,
            power=args.power, floor=args.floor, bandwidth=args.bandwidth)
    # Verify the production helper agrees on the uniform shape (a guard
    # against the two implementations drifting apart).
    if args.shape == "uniform" and not args.subset_prior and args.cap <= 0:
        production = _bounded_prior_weights(y, args.horizon, bandwidth=args.bandwidth,
                                            power=args.power, floor=args.floor)
        if not np.allclose(production, weights, rtol=1e-12, atol=1e-12):
            raise SystemExit("research weights disagree with the production helper")

    history = []
    for epoch in range(1, int(args.pretrain_epochs) + 1):
        model.fit(ctx["train"]["x"], target, 1)
        history.append({"epoch": epoch, "phase": "pretrain"})
    for epoch in range(1, int(args.epochs) + 1):
        model.fit(ctx["train"]["x"], target, 1, sample_weight=weights)
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = cw._metrics(ctx, model.predict(ctx["valid"]["x"]))
            metrics["epoch"] = epoch
            history.append(metrics)

    pred = model.predict(ctx["valid"]["x"])
    metrics = cw._metrics(ctx, pred)
    metrics["uniform_protocol"] = uniform_protocol(ctx, pred)
    result = {
        "config": vars(args),
        "parent": parent,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "metrics": metrics,
        "history": history,
    }
    print(json.dumps({k: result[k] for k in ("config", "weight_min", "weight_max")}, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in
                      ("full_rmse", "full_le195_rmse", "proxy_rmse", "uniform_protocol")},
                     ensure_ascii=False))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
