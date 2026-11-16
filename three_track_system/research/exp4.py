# -*- coding: utf-8 -*-
"""Experiment 4: fast closed-form evaluation of per-bearing healthy-baseline normalisation."""
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
        for ch in ("horizontal", "vertical"):
            for k in f[ch]:
                v.append(float(f[ch][k]))
        xs.append(v); ys.append(float(r["answer"])); ids.append(r["id"])
    return np.array(xs), np.array(ys), ids


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def norm_feats(x, groups, k, mode):
    out = np.empty_like(x)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        head = idx[:min(k, len(idx))]
        base = np.median(x[head], axis=0)
        mad = np.median(np.abs(x[head] - base), axis=0) * 1.4826
        sd = np.maximum(mad, 1e-9)
        if mode == "ratio":
            out[idx] = x[idx] / np.maximum(np.abs(base), 1e-9)
        elif mode == "z":
            out[idx] = (x[idx] - base) / sd
        elif mode == "logratio":
            out[idx] = np.log(np.maximum(np.abs(x[idx]), 1e-12)) - np.log(np.maximum(np.abs(base), 1e-12))
        elif mode == "ratio+z":
            out[idx] = np.clip((x[idx] - base) / sd, -10, 10)
    return out


def augment(F):
    """add log of absolute values so magnitude info survives"""
    return np.c_[F, np.log(np.maximum(np.abs(F), 1e-9))]


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1
    Z = np.clip((X - mu) / sd, -10, 10)
    ym, ys = y.mean(), max(y.std(), 1e-9)
    A = np.c_[Z, np.ones(len(Z))]
    reg = np.eye(A.shape[1]) * alpha; reg[-1, -1] = 0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ ((y - ym) / ys))
    return mu, sd, ym, ys, coef


def ridge_pred(m, X):
    mu, sd, ym, ys, coef = m
    Z = np.clip((X - mu) / sd, -10, 10)
    return np.maximum(0, (np.c_[Z, np.ones(len(Z))] @ coef) * ys + ym)


if __name__ == "__main__":
    xtr, ytr, idtr = load(BASE + r"\train.jsonl")
    xva, yva, idva = load(BASE + r"\validation.jsonl")
    gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
    gva = np.array([i.rsplit("_s", 1)[0] for i in idva])
    folds = sorted(set(gtr))

    for k in (5, 10, 20, 30):
        for mode in ("ratio", "z", "logratio"):
            Ntr = norm_feats(xtr, gtr, k, mode)
            Nva = norm_feats(xva, gva, k, mode)
            best = (1e18, None, None)
            for alpha in (1, 10, 100, 1000):
                pred = np.zeros(len(ytr))
                for b in folds:
                    m = gtr == b
                    pred[m] = ridge_pred(ridge_fit(Ntr[~m], ytr[~m], alpha), Ntr[m])
                r = rmse(pred, ytr)
                if r < best[0]:
                    best = (r, alpha, pred)
            _, alpha, pred = best
            v = ridge_pred(ridge_fit(Ntr, ytr, alpha), Nva)
            per = {b: rmse(pred[gtr == b], ytr[gtr == b]) for b in folds}
            print(f"k={k:2d} {mode:9s} a={alpha:<5} LOO={best[0]:7.1f} | val={rmse(v,yva):7.1f} mae={np.mean(np.abs(v-yva)):7.1f} | "
                  + " ".join(f"{b.split('_')[1]}:{per[b]:.0f}" for b in folds))
