# -*- coding: utf-8 -*-
"""Profile C (turbofan RUL): sample-weighting experiments on the *hidden* protocol.

Why this exists
---------------
Profile C's public validation split contains **every row of every complete
unit**, but the platform's hidden split is the official CMAPSS protocol:
one row per unit, taken at the last observed cycle before truncation.  The
two rows of the same engine at RUL=400 and RUL=80 are weighted equally by
the full-validation RMSE, while only the latter exists in the hidden split.
A training loss that fits the full validation set therefore spends most of
its mass on a lifecycle region the hidden metric never sees.

The experiment here asks a single question: **does re-weighting the training
rows so that their remaining-life distribution matches a truncated-test
distribution lower the hidden-protocol RMSE?**  Every variant trains with the
same architecture, seed and optimiser; only the per-row loss mass differs.

The hidden-protocol proxy
-------------------------
``proxy_rmse`` is the expected RMSE of one row per validation engine, with
each engine's truncation horizon drawn from that subset's public CMAPSS
``RUL_FD00x.txt`` distribution and the four subsets weighted by their hidden
engine counts.  It is a *proxy*: the private evaluation package is a
different dataset.  The point is to measure the direction and size of the
protocol gap, not to claim the exact hidden score.

Two families of weights are compared:

* ``imp*`` - importance weights ``p_test(RUL) / p_train(RUL)``.  With the
  empirical CMAPSS test distribution these are the RMSE-optimal weights for
  the hidden protocol; the ``prior`` variants replace the empirical density
  with a generic bounded prior (uniform / triangular / exponential) so the
  result does not depend on the exact public test labels.
* ``deg*`` / ``surv*`` / ``hard*`` - simpler late-life re-weightings.
* ``cap*`` - train against a capped target (a classical C-MAPSS recipe).

(Important: the hidden-protocol weights are a *training-side* change.  The
feature schema, the forward pass and the checkpoint format are untouched, so
every candidate stays a deployable ``contest.cpu-mlp.v1`` checkpoint that
needs only ``inputs`` at prediction time.)

Run::

    python -m research.tracks.c_weight --epochs 30 --seeds 42,7,123
    python -m research.tracks.c_weight --only control,deg1,imp --seeds 42

The data root is resolved from ``--package`` / ``CPU_C_PACKAGE`` / the usual
sibling directories; the CMAPSS ``RUL_FD00x.txt`` priors live in
``research/data/cmapss``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT.parent / "packages" / "cpu-c-guide-20261110-v1"
RUL_DIR = ROOT / "research" / "data" / "cmapss"


def _package(path=None):
    """Locate the official C package, without hard-coding one machine."""
    candidates = [path, os.environ.get("CPU_C_PACKAGE"), DEFAULT_PACKAGE,
                  ROOT.parent / "数据源" / "cpu-c-guide-20261110-v1 (1)"]
    for candidate in candidates:
        if candidate and (pathlib.Path(candidate) / "public" / "train.jsonl").is_file():
            return pathlib.Path(candidate)
    raise FileNotFoundError("could not locate cpu-c-guide-20261110-v1; pass --package")


def load(package=None):
    """Read the public splits and build the evaluation caches."""
    pkg = _package(package)
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from baselines.data import read_dataset

    train = read_dataset(pkg / "public" / "train.jsonl", "regression")
    valid = read_dataset(pkg / "public" / "validation.jsonl", "regression", None, train["paths"])

    def meta_rows(name):
        rows = [json.loads(line) for line in (pkg / "public" / name).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        subset = np.array([row["meta"]["source_subset"] for row in rows])
        unit = np.array([f"{row['meta']['source_subset']}#{row['meta']['unit']}" for row in rows])
        cycle = np.array([row["meta"]["cycle"] for row in rows], dtype=np.float64)
        return subset, unit, cycle

    tsub, tunit, _ = meta_rows("train.jsonl")
    vsub, vunit, vcyc = meta_rows("validation.jsonl")
    subsets = sorted(set(vsub.tolist()))

    rul = {s: np.loadtxt(RUL_DIR / f"RUL_{s}.txt", dtype=int).ravel() for s in ("FD001", "FD002", "FD003", "FD004")}
    hidden_count = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
    maxh = int(max(r.max() for r in rul.values()))

    last = {u: float(vcyc[vunit == u].max()) for u in sorted(set(vunit.tolist()))}
    engines = {s: [u for u in last if u.startswith(s + "#")] for s in subsets}
    rows_at = {u: {} for u in last}
    for _u, members in ((u, np.nonzero(vunit == u)[0]) for u in last):
        for h in range(maxh + 1):
            target = last[_u] - h
            k = int(members[np.argmin(np.abs(vcyc[members] - target))])
            if abs(vcyc[k] - target) <= 0.5:
                rows_at[_u][h] = k
    ph = {s: np.bincount(rul[s], minlength=maxh + 1).astype(float) / len(rul[s]) for s in rul}

    umax = {}
    for unit, value in zip(tunit, train["y"]):
        umax[unit] = max(umax.get(unit, 0.0), float(value))
    unit_max = np.array([umax[u] for u in tunit])
    units = np.array(sorted(set(tunit.tolist())))
    unit_counts = np.array([np.sum(tunit == u) for u in units])

    def smooth_hist(values, bw=8.0):
        grid = np.arange(0, maxh + 1, dtype=float)
        hist = np.bincount(np.clip(np.round(values).astype(int), 0, maxh), minlength=maxh + 1).astype(float)
        kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / bw) ** 2)
        kernel /= kernel.sum(axis=1, keepdims=True)
        out = kernel @ hist
        return out / out.sum()

    p_train = smooth_hist(train["y"])
    p_hidden = smooth_hist(np.concatenate([rul[s] for s in rul]))
    importance = np.clip(p_hidden / (p_train + 1e-12), 0.02, 20.0)
    importance /= importance.mean()
    importance_sub = np.zeros(len(train["y"]))
    for s in ("FD001", "FD002", "FD003", "FD004"):
        p_train_s = smooth_hist(train["y"][tsub == s])
        ratio = np.clip(ph[s] / (p_train_s + 1e-12), 0.02, 20.0)
        mask = tsub == s
        importance_sub[mask] = ratio[np.clip(np.round(train["y"][mask]).astype(int), 0, maxh)]
    importance_sub /= importance_sub.mean()

    return {
        "pkg": pkg, "train": train, "valid": valid, "subsets": subsets,
        "rul": rul, "hidden_count": hidden_count, "maxh": maxh,
        "last": last, "engines": engines, "rows_at": rows_at, "ph": ph,
        "unit_max": unit_max, "units": units, "unit_counts": unit_counts, "tunit": tunit,
        "importance": importance, "importance_sub": importance_sub,
        "p_train": p_train, "p_hidden": p_hidden, "tsub": tsub,
    }


def _metrics(ctx, pred):
    yv = ctx["valid"]["y"]
    err = pred - yv
    full = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ratio = np.maximum(np.abs(yv), 1.0)
    penalty = np.where(err > 0, 5.0 * np.maximum(err, 0.0) / ratio, np.maximum(-err, 0.0) / ratio)

    per = {}
    for s in ctx["subsets"]:
        squared = []
        for u in ctx["engines"][s]:
            p = ctx["ph"][s]
            num = den = 0.0
            for h, k in ctx["rows_at"][u].items():
                if p[h] <= 0:
                    continue
                e = float(pred[k] - yv[k])
                num += p[h] * e * e
                den += p[h]
            if den > 0:
                squared.append(num / den)
        per[s] = float(np.sqrt(np.mean(squared)))
    pooled = float(np.sqrt(sum(ctx["hidden_count"][s] * per[s] ** 2 for s in ctx["subsets"])
                           / sum(ctx["hidden_count"].values())))
    le = yv <= 195
    full_le = float(np.sqrt(np.mean(err[le] ** 2)))
    horizon = {}
    for h in (0, 10, 25, 50, 75, 100, 125, 150, 175, 195):
        idx = np.array([ctx["rows_at"][u][h] for u in ctx["last"] if h in ctx["rows_at"][u]], dtype=int)
        horizon[str(h)] = float(np.sqrt(np.mean((pred[idx] - yv[idx]) ** 2)))
    return {"full_rmse": full, "full_mae": mae, "nasa_mean": float(penalty.mean()),
            "full_le195_rmse": full_le, "proxy_rmse": pooled, "proxy_per_subset": per,
            "horizon_rmse": horizon}


def weights_for(ctx, kind, **kw):
    y = ctx["train"]["y"]
    maxh = ctx["maxh"]
    index = np.clip(np.round(y).astype(int), 0, maxh)
    power = float(kw.get("power", 1.0))
    if kind == "none":
        return None
    if kind == "deg":
        alpha = float(kw.get("alpha", 1.0))
        w = 1.0 + alpha * (1.0 - np.clip(y / ctx["unit_max"], 0, 1))
    elif kind == "imp":
        w = ctx["importance"][index]
    elif kind == "density":
        w = ctx["p_hidden"][index]
    elif kind == "surv":
        h = float(kw.get("H", 150.0)); floor = float(kw.get("floor", 0.1)); scale = float(kw.get("scale", 75.0))
        w = np.maximum(1.0 / (1.0 + np.maximum(0.0, y - h) / scale), floor)
    elif kind == "unit":
        w = np.array([1.0 / ctx["unit_counts"][np.searchsorted(ctx["units"], u)] for u in ctx["tunit"]])
    elif kind == "impunit":
        w = ctx["importance"][index] * np.array(
            [1.0 / ctx["unit_counts"][np.searchsorted(ctx["units"], u)] for u in ctx["tunit"]])
    elif kind == "impsub":
        w = ctx["importance_sub"]
    elif kind in ("prior", "hard"):
        h = float(kw.get("H", 195.0)); shape = kw.get("shape", "uniform")
        tau = float(kw.get("tau", 80.0)); floor = float(kw.get("floor", 0.02))
        if kind == "hard":
            w = np.where(y <= h, 1.0, max(floor, 0.0)) ** power
            return w / w.mean()
        grid = np.arange(0, maxh + 1, dtype=float)
        if shape == "uniform":
            p = (grid <= h).astype(float)
        elif shape == "tri":
            p = np.where(grid <= h, h - grid + 1.0, 0.0)
        elif shape == "exp":
            p = np.where(grid <= h, np.exp(-grid / tau), 0.0)
        else:
            raise ValueError(shape)
        p /= p.sum()
        w = np.clip(p / (ctx["p_train"] + 1e-12), floor, 20.0)[index]
    elif kind == "degimp":
        alpha = float(kw.get("alpha", 1.0))
        w = (1.0 + alpha * (1.0 - np.clip(y / ctx["unit_max"], 0, 1))) * ctx["importance"][index]
    else:
        raise ValueError(kind)
    w = np.asarray(w, dtype=float) ** power
    return w / w.mean()


VARIANTS = [
    ("control", dict(w="none")),
    ("deg.5", dict(w="deg", alpha=0.5)),
    ("deg1", dict(w="deg", alpha=1.0)),
    ("deg2", dict(w="deg", alpha=2.0)),
    ("imp", dict(w="imp")),
    ("density", dict(w="density")),
    ("surv150", dict(w="surv", H=150, scale=75, floor=0.1)),
    ("surv100", dict(w="surv", H=100, scale=50, floor=0.05)),
    ("cap125", dict(w="none", cap=125)),
    ("cap150", dict(w="none", cap=150)),
    ("cap200", dict(w="none", cap=200)),
    ("cap150+deg1", dict(w="deg", alpha=1.0, cap=150)),
    ("cap150+imp", dict(w="imp", cap=150)),
    ("impunit", dict(w="impunit")),
    ("imp0.5", dict(w="imp", power=0.5)),
    ("imp0.75", dict(w="imp", power=0.75)),
    ("density0.5", dict(w="density", power=0.5)),
    ("surv150p0.5", dict(w="surv", H=150, scale=75, floor=0.1, power=0.5)),
    ("impsub", dict(w="impsub")),
    ("impsub0.5", dict(w="impsub", power=0.5)),
    ("cap150+deg2", dict(w="deg", alpha=2.0, cap=150)),
    ("cap150+imp0.5", dict(w="imp", power=0.5, cap=150)),
    ("impunif195", dict(w="prior", shape="uniform", H=195)),
    ("impunif150", dict(w="prior", shape="uniform", H=150)),
    ("imptri195", dict(w="prior", shape="tri", H=195)),
    ("impexp195", dict(w="prior", shape="exp", H=195, tau=80)),
    ("hard195", dict(w="hard", H=195, floor=0.01)),
    ("hard150", dict(w="hard", H=150, floor=0.01)),
    ("deg2+imp0.5", dict(w="degimp", alpha=2.0)),
    ("unif195p.5", dict(w="prior", shape="uniform", H=195, power=0.5)),
    ("unif195p.75", dict(w="prior", shape="uniform", H=195, power=0.75)),
    ("exp195p.5", dict(w="prior", shape="exp", H=195, tau=80, power=0.5)),
    ("hard195p.5", dict(w="hard", H=195, floor=0.1, power=0.5)),
]


def run(ctx, name, spec, *, epochs=30, lr=0.004, batch=256, verbose=True):
    from contest_baselines.model import Model

    epochs = int(spec.get("epochs", epochs)); lr = float(spec.get("lr", lr)); batch = int(spec.get("batch", batch))
    cap = spec.get("cap"); seed = int(spec.get("seed", 42))
    model = Model.load(ctx["pkg"] / "checkpoint.npz")
    model.meta["learning_rate"] = lr
    model.meta["batch_size"] = batch
    model.meta["seed"] = seed
    model.validate()
    weights = weights_for(ctx, spec["w"], **spec)
    y_train = ctx["train"]["y"] if cap is None else np.minimum(ctx["train"]["y"], float(cap))
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        loss = model.fit(ctx["train"]["x"], y_train, 1, sample_weight=weights)[0]
        metrics = _metrics(ctx, model.predict(ctx["valid"]["x"]))
        metrics["epoch"] = epoch
        metrics["loss"] = loss
        history.append(metrics)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  {name} ep{epoch:02d} full {metrics['full_rmse']:.3f} proxy {metrics['proxy_rmse']:.3f}",
                  flush=True)
    seconds = time.time() - started
    best_proxy = min(history, key=lambda row: row["proxy_rmse"])
    best_full = min(history, key=lambda row: row["full_rmse"])
    result = {"variant": name, "spec": spec, "epochs": epochs, "lr": lr, "batch": batch, "cap": cap,
              "seconds": round(seconds, 1), "final": history[-1], "best_proxy": best_proxy,
              "best_full": best_full, "history": history}
    if verbose:
        print(f"{name:14s} full {history[-1]['full_rmse']:.3f} (best {best_full['full_rmse']:.3f}@{best_full['epoch']}) "
              f"proxy {history[-1]['proxy_rmse']:.3f} (best {best_proxy['proxy_rmse']:.3f}@{best_proxy['epoch']}) "
              f"{seconds:.0f}s", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "tracks" / "c_weight.json"))
    args = parser.parse_args(argv)
    ctx = load(args.package or None)
    results = []
    for name, spec in VARIANTS:
        if args.only and name not in args.only.split(","):
            continue
        for seed in (int(v) for v in args.seeds.split(",")):
            label = f"{name}@s{seed}"
            print(f"== {label} ==", flush=True)
            results.append(run(ctx, label, {**spec, "seed": seed}, epochs=args.epochs))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
