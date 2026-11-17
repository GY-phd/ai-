# -*- coding: utf-8 -*-
"""Diagnostics: how much is learnable across bearings? Leave-one-bearing-out."""
import json
import numpy as np

BASE = r"D:\GY\deepseek\cpu-b-rul-20261110-v1\public"


def load(path):
    ids, xs, ys = [], [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        f = r["inputs"]["features"]
        v = []
        names = []
        for ch in ("horizontal", "vertical"):
            for k in f[ch]:
                v.append(float(f[ch][k]))
                names.append(ch[0] + "_" + k)
        xs.append(v)
        ys.append(float(r["answer"]))
        ids.append(r["id"])
    return np.array(xs), np.array(ys), ids, names


xtr, ytr, idtr, names = load(BASE + r"\train.jsonl")
xva, yva, idva, _ = load(BASE + r"\validation.jsonl")
gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
gva = np.array([i.rsplit("_s", 1)[0] for i in idva])

print("feature names:", names)
print()
print("== per-bearing Spearman correlation (top |rho|) ==")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


for b in np.unique(gtr):
    m = gtr == b
    rho = np.array([spearman(xtr[m, j], ytr[m]) for j in range(xtr.shape[1])])
    top = np.argsort(-np.abs(rho))[:5]
    print(f"{b:16s} n={m.sum():5d} " + ", ".join(f"{names[j]}={rho[j]:+.2f}" for j in top))

print()
print("== per-bearing best single feature (linear fit, oracle upper bound) ==")
for b in np.unique(gtr):
    m = gtr == b
    X, y = xtr[m], ytr[m]
    best = (1e18, None)
    for j in range(X.shape[1]):
        A = np.c_[X[:, j], np.ones(len(X))]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = np.clip(A @ coef, 0, None)
        r = float(np.sqrt(np.mean((pred - y) ** 2)))
        if r < best[0]:
            best = (r, names[j])
    print(f"{b:16s} oracle-linear rmse={best[0]:7.1f} ({best[1]})  const-zero={float(np.sqrt((y**2).mean())):7.1f}")

print()
print("== leave-one-bearing-out: how well does a global model transfer? ==")


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1
    Z = np.clip((X - mu) / sd, -10, 10)
    ym, ys = y.mean(), max(y.std(), 1e-9)
    A = np.c_[Z, np.ones(len(Z))]
    n, d = A.shape
    reg = np.eye(d) * alpha
    reg[-1, -1] = 0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ ((y - ym) / ys))
    return (mu, sd, ym, ys, coef)


def ridge_pred(model, X):
    mu, sd, ym, ys, coef = model
    Z = np.clip((X - mu) / sd, -10, 10)
    A = np.c_[Z, np.ones(len(Z))]
    return np.maximum(0, A @ coef * ys + ym)


for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
    preds = np.zeros(len(ytr))
    for b in np.unique(gtr):
        m = gtr == b
        mod = ridge_fit(xtr[~m], ytr[~m], alpha)
        preds[m] = ridge_pred(mod, xtr[m])
    print(f"  ridge alpha={alpha:8.1f}  LOO rmse={np.sqrt(np.mean((preds-ytr)**2)):8.2f}  mae={np.mean(np.abs(preds-ytr)):8.2f}")

preds = np.zeros(len(ytr))
for b in np.unique(gtr):
    m = gtr == b
    preds[m] = ytr[~m].mean()
print(f"  global-mean          LOO rmse={np.sqrt(np.mean((preds-ytr)**2)):8.2f}  mae={np.mean(np.abs(preds-ytr)):8.2f}")

preds = np.zeros(len(ytr))
for b in np.unique(gtr):
    m = gtr == b
    preds[m] = 0.0
print(f"  zero                 LOO rmse={np.sqrt(np.mean((preds-ytr)**2)):8.2f}")

preds = np.zeros(len(ytr))
for b in np.unique(gtr):
    m = gtr == b
    preds[m] = np.median(ytr[~m])
print(f"  global-median        LOO rmse={np.sqrt(np.mean((preds-ytr)**2)):8.2f}")
