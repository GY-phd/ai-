# -*- coding: utf-8 -*-
"""Two questions about profile C, answered on the public data.

Question 1 - the conjunction rule
    The platform's 2026-11-23 run rejected several candidates whose *full*
    split RMSE improved, because the protocol-aligned slice (``late_rmse``)
    got slightly worse.  Is a tolerance on the slice the right fix, or
    should the two readings be combined into one score?

Question 2 - the +10.06% public-to-hidden gap
    The deployed model scored 51.111 on the public split and 56.253 on the
    hidden split, while the static baseline scored 52.869 public.  The
    hidden pack is a *different* dataset, so the question is which
    *offline* protocol predicts the hidden ranking best.

The experiment
--------------
The official C package ships the public split and the shipped static
checkpoint.  The hidden pack is unavailable, so every protocol below is a
*proxy*; the point is to compare protocols against each other, not to
predict the hidden number.

Three evaluation protocols are computed for the same model:

``public_full``
    RMSE over every row of the public validation split.  This is what the
    runtime currently optimises.

``public_truncated``
    RMSE over one row per validation engine, taken at the last observed
    cycle (the official CMAPSS test protocol shape).  This is the
    protocol the slice is supposed to approximate.

``leave-one-engine-out``
    Rotate over the public validation engines: train on the public
    *training* engines plus all validation engines except one, evaluate on
    the held-out one.  Averaged over the folds this measures how the model
    behaves on an engine it has never seen, without ever touching the
    hidden pack.

A protocol is only useful if it *ranks* candidates the way the hidden
protocol does.  The ranking correlation against ``public_truncated`` (the
closest available stand-in for the hidden protocol) is the number that
decides which protocol to trust.

Run::

    python -m research.tracks.c_protocol --epochs 30 --seeds 42,7
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "art" / "repo-inspect" / "cpu-c-guide-20261110-v1"


# ------------------------------------------------------------------ loading
def load(package=None):
    """Read the public splits and the engine grouping."""
    pkg = pathlib.Path(package) if package else PACKAGE
    for path in (str(pkg), str(pkg / "baselines")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from baselines.data import read_dataset

    train = read_dataset(pkg / "public" / "train.jsonl", "regression")
    valid = read_dataset(pkg / "public" / "validation.jsonl", "regression", None, train["paths"])

    def meta(name):
        rows = [json.loads(line) for line in (pkg / "public" / name).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        subset = np.array([r["meta"]["source_subset"] for r in rows])
        unit = np.array([f"{r['meta']['source_subset']}#{r['meta']['unit']}" for r in rows])
        cycle = np.array([r["meta"]["cycle"] for r in rows], dtype=np.float64)
        return subset, unit, cycle

    tsub, tunit, tcycle = meta("train.jsonl")
    vsub, vunit, vcycle = meta("validation.jsonl")

    # The public validation split carries complete engines, so the "last
    # observed cycle" of a validation engine is simply its max cycle.
    v_engines = sorted(set(vunit.tolist()))
    t_engines = sorted(set(tunit.tolist()))
    return {
        "pkg": pkg, "train": train, "valid": valid,
        "tsub": tsub, "tunit": tunit, "tcycle": tcycle,
        "vsub": vsub, "vunit": vunit, "vcycle": vcycle,
        "t_engines": t_engines, "v_engines": v_engines,
    }


def engine_index(ctx):
    """Index of every row's engine, plus the last-cycle row per engine."""
    vunit = ctx["vunit"]
    vcycle = ctx["vcycle"]
    last_row = {}
    for engine in ctx["v_engines"]:
        members = np.nonzero(vunit == engine)[0]
        last_row[engine] = int(members[np.argmax(vcycle[members])])
    return last_row


# ------------------------------------------------------------------ scoring
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def nasa(y, p):
    y = np.asarray(y, dtype=float)
    err = np.asarray(p, dtype=float) - y
    ratio = np.maximum(np.abs(y), 1.0)
    pen = np.where(err > 0, 5.0 * np.maximum(err, 0.0) / ratio, np.maximum(-err, 0.0) / ratio)
    return float(pen.sum()), float(pen.mean())


def score_all(ctx, pred_valid, last_row):
    """Score one prediction vector under every protocol."""
    y = ctx["valid"]["y"]
    err = np.asarray(pred_valid, dtype=float) - y

    full = rmse(y, pred_valid)
    le200 = y <= 200
    le195 = y <= 195
    late200 = rmse(y[le200], pred_valid[le200])
    late195 = rmse(y[le195], pred_valid[le195])

    # One row per validation engine at its last observed cycle.
    idx = np.array([last_row[e] for e in ctx["v_engines"]], dtype=int)
    trunc = rmse(y[idx], pred_valid[idx])

    # One row per engine at a fixed truncation horizon, averaged over the
    # public CMAPSS test-shaped distribution of horizons.
    horizons = (0, 10, 25, 50, 75, 100, 125, 150, 175, 195)
    per_h = {}
    for h in horizons:
        rows = []
        for engine in ctx["v_engines"]:
            members = np.nonzero(ctx["vunit"] == engine)[0]
            target = ctx["vcycle"][members].max() - h
            k = int(members[np.argmin(np.abs(ctx["vcycle"][members] - target))])
            if abs(ctx["vcycle"][k] - target) <= 0.5:
                rows.append(k)
        rows = np.array(rows, dtype=int)
        per_h[str(h)] = rmse(y[rows], pred_valid[rows])

    total, mean_pen = nasa(y, pred_valid)
    return {
        "public_full_rmse": full,
        "public_late200_rmse": late200,
        "public_late195_rmse": late195,
        "truncated_rmse": trunc,
        "horizon_rmse": per_h,
        "nasa_total": total,
        "nasa_mean": mean_pen,
        "mae": float(np.mean(np.abs(err))),
    }


# ------------------------------------------------------------------ training
def make_weights(ctx, spec):
    """Per-row training weights for the candidate recipes."""
    y = ctx["train"]["y"]
    tunit = ctx["tunit"]
    kind = spec.get("w", "none")
    unit_max = {}
    for unit, value in zip(tunit, y):
        unit_max[unit] = max(unit_max.get(unit, 0.0), float(value))
    umax = np.array([unit_max[u] for u in tunit])
    counts = {}
    for unit in tunit:
        counts[unit] = counts.get(unit, 0) + 1
    ucount = np.array([counts[u] for u in tunit])

    if kind == "none":
        return None
    if kind == "surv":
        h = float(spec.get("H", 150.0))
        floor = float(spec.get("floor", 0.05))
        scale = float(spec.get("scale", 75.0))
        w = np.maximum(1.0 / (1.0 + np.maximum(0.0, y - h) / scale), floor)
    elif kind == "deg":
        alpha = float(spec.get("alpha", 1.0))
        w = 1.0 + alpha * (1.0 - np.clip(y / umax, 0, 1))
    elif kind == "unit":
        w = 1.0 / ucount
    else:
        raise ValueError(kind)
    w = np.asarray(w, dtype=float) ** float(spec.get("power", 1.0))
    return w / w.mean()


def train_candidate(ctx, spec, *, epochs=30, lr=0.004, batch=256, seed=42):
    """Train one candidate from the official checkpoint and score it.

    ``contest_baselines.model.Model`` is used rather than the shipped
    baseline's, because it is the same implementation the platform runs and
    it accepts ``sample_weight``.  It reads and writes the official
    ``contest.cpu-mlp.v1`` checkpoint format, so the candidates stay
    deployable.
    """
    from contest_baselines.model import Model

    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    model.meta["learning_rate"] = lr
    model.meta["batch_size"] = batch
    model.meta["seed"] = seed
    model.validate()
    weights = make_weights(ctx, spec)
    cap = spec.get("cap")
    y_train = ctx["train"]["y"] if cap is None else np.minimum(ctx["train"]["y"], float(cap))
    for _ in range(int(epochs)):
        model.fit(ctx["train"]["x"], y_train, 1, sample_weight=weights)
    return model


# ------------------------------------------------------------------ the study
def variants():
    """The candidate recipes the closed loop actually produced.

    The names and shapes come from the platform's own run report, so the
    study measures the candidates that were really on the table rather
    than a synthetic sweep.
    """
    return [
        ("control", dict(w="none")),
        ("surv150", dict(w="surv", H=150, scale=75, floor=0.05)),
        ("surv150f10", dict(w="surv", H=150, scale=75, floor=0.10)),
        ("surv100", dict(w="surv", H=100, scale=50, floor=0.05)),
        ("surv195", dict(w="surv", H=195, scale=75, floor=0.05)),
        ("deg1", dict(w="deg", alpha=1.0)),
        ("deg2", dict(w="deg", alpha=2.0)),
        ("deg0.5", dict(w="deg", alpha=0.5)),
        ("deg2p0.5", dict(w="deg", alpha=2.0, power=0.5)),
        ("cap125", dict(w="none", cap=125)),
        ("cap150", dict(w="none", cap=150)),
        ("cap200", dict(w="none", cap=200)),
        ("unit", dict(w="unit")),
        ("surv150deg2", dict(w="surv", H=150, scale=75, floor=0.05)),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_protocol.json"))
    args = parser.parse_args(argv)

    ctx = load(args.package or None)
    last_row = engine_index(ctx)
    print("public train rows %d / %d engines; validation %d rows / %d engines"
          % (len(ctx["train"]["y"]), len(ctx["t_engines"]),
             len(ctx["valid"]["y"]), len(ctx["v_engines"])), flush=True)

    results = []
    wanted = set(args.only.split(",")) if args.only else None
    for name, spec in variants():
        if wanted and name not in wanted:
            continue
        for seed in (int(s) for s in args.seeds.split(",")):
            started = time.time()
            model = train_candidate(ctx, spec, epochs=args.epochs, seed=seed)
            scored = score_all(ctx, model.predict(ctx["valid"]["x"]), last_row)
            scored.update(variant=name, spec=spec, seed=seed,
                          seconds=round(time.time() - started, 1))
            results.append(scored)
            print("%-12s s%-4d full %8.4f  late200 %8.4f  trunc %8.4f  %.0fs"
                  % (name, seed, scored["public_full_rmse"], scored["public_late200_rmse"],
                     scored["truncated_rmse"], scored["seconds"]), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
