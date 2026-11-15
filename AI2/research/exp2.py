# -*- coding: utf-8 -*-
"""Experiment 2: feature engineering + leave-one-bearing-out selection."""
import json
import numpy as np

BASE = r"D:\GY\deepseek\cpu-b-rul-20261110-v1\public"
CH = ("horizontal", "vertical")


def load(path):
    ids, xs, ys = [], [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        f = r["inputs"]["features"]
        v = []
        for ch in CH:
            for k in f[ch]:
                v.append(float(f[ch][k]))
        xs.append(v)
        ys.append(float(r["answer"]))
        ids.append(r["id"])
    return np.array(xs), np.array(ys), ids


def build(x, mode):
    """mode: raw | log | sf | log+sf. Returns (features, names)."""
    n = x.shape[0]
    cols, names = [], []

    def add(name, arr):
        cols.append(arr.reshape(n, -1))
        for i in range(arr.reshape(n, -1).shape[1]):
            names.append(f"{name}_{i}" if arr.reshape(n, -1).shape[1] > 1 else name)

    for c, ch in enumerate(CH):
        base = c * 17
        g = x[:, base:base + 17]
        rms, peak, ptp, mean_abs, kurt, skew, crest, shape, impulse, margin, sc, srms, skurt, dfreq, b1, b2, b3 = g.T
        if "raw" in mode:
            add(f"{ch}_rms", rms); add(f"{ch}_peak", peak); add(f"{ch}_kurt", kurt)
        if "log" in mode:
            for nm, arr in (("lrms", rms), ("lpeak", peak), ("lptp", ptp), ("lmean", mean_abs),
                            ("lsc", sc), ("lsrms", srms), ("lskurt", skurt),
                            ("lb1", b1), ("lb2", b2), ("lb3", b3)):
                add(f"{ch}_{nm}", np.log(np.maximum(arr, 1e-9)))
        if "sf" in mode:
            tot = np.maximum(b1 + b2 + b3, 1e-12)
            add(f"{ch}_kurt", kurt); add(f"{ch}_skew", skew)
            add(f"{ch}_crest", crest); add(f"{ch}_shape", shape)
            add(f"{ch}_impulse", impulse); add(f"{ch}_margin", margin)
            add(f"{ch}_r_b1", b1 / tot); add(f"{ch}_r_b2", b2 / tot); add(f"{ch}_r_b3", b3 / tot)
            add(f"{ch}_scn", sc / 12800.0); add(f"{ch}_dfn", dfreq / 12800.0)
            add(f"{ch}_lskurt", np.log(np.maximum(skurt, 1e-9)))
    return np.concatenate(cols, axis=1), names


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


class MLP:
    def __init__(self, d, hidden, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1.0 / d), (d, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, 1)); self.b2 = np.zeros(1)
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        self.t = 0

    def freeze_from(self, other):
        """copy the overlapping (features x units) block of a narrower model"""
        h = other.W1.shape[1]
        self.W1[:, :h] = other.W1
        self.b1[:h] = other.b1
        self.W2[:h, :] = other.W2

    def fwd(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def step(self, xb, yb, lr):
        h, o = self.fwd(xb)
        d = (2 * (o[:, 0] - yb) / len(xb))[:, None]
        gW2 = h.T @ d; gb2 = d.sum(0)
        dh = (d @ self.W2.T) * (1 - h ** 2)
        gW1 = xb.T @ dh; gb1 = dh.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        sc = min(1.0, 5.0 / max(norm, 1e-12))
        self.t += 1
        for i, g in enumerate(grads):
            g *= sc
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * g ** 2
            p = [self.W1, self.b1, self.W2, self.b2][i]
            p -= lr * (self.m[i] / (1 - .9 ** self.t)) / (np.sqrt(self.v[i] / (1 - .999 ** self.t)) + 1e-8)


def fit_mlp(X, y, Xv_list, epochs=200, lr=3e-3, bs=128, hidden=64, seed=42):
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1
    Z = np.clip((X - mu) / sd, -10, 10)
    ym, ys = y.mean(), max(y.std(), 1e-9)
    yt = (y - ym) / ys
    net = MLP(Z.shape[1], hidden, seed)
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Z))
        cur = lr * 0.5 * (1 + np.cos(np.pi * (ep - 1) / epochs))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            net.step(Z[idx], yt[idx], cur)
    outs = []
    for Xv in Xv_list:
        Zv = np.clip((Xv - mu) / sd, -10, 10)
        outs.append(np.maximum(0, net.fwd(Zv)[1][:, 0] * ys + ym))
    return outs, net


def fit_ridge(X, y, Xv_list, alpha):
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1
    Z = np.clip((X - mu) / sd, -10, 10)
    ym, ys = y.mean(), max(y.std(), 1e-9)
    A = np.c_[Z, np.ones(len(Z))]
    reg = np.eye(A.shape[1]) * alpha; reg[-1, -1] = 0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ ((y - ym) / ys))
    outs = []
    for Xv in Xv_list:
        Zv = np.clip((Xv - mu) / sd, -10, 10)
        outs.append(np.maximum(0, (np.c_[Zv, np.ones(len(Zv))] @ coef) * ys + ym))
    return outs


if __name__ == "__main__":
    xtr, ytr, idtr = load(BASE + r"\train.jsonl")
    xva, yva, idva = load(BASE + r"\validation.jsonl")
    gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
    folds = sorted(set(gtr))

    for mode in ("raw", "log", "sf", "log+sf"):
        Ftr, names = build(xtr, mode)
        Fva, _ = build(xva, mode)
        # LOO ridge
        best = (1e18, None)
        for alpha in (0.3, 1, 3, 10, 30, 100, 300, 1000):
            pred = np.zeros(len(ytr))
            for b in folds:
                m = gtr == b
                pred[m] = fit_ridge(Ftr[~m], ytr[~m], [Ftr[m]], alpha)[0]
            r = rmse(pred, ytr)
            if r < best[0]:
                best = (r, alpha)
        # LOO mlp
        predm = np.zeros(len(ytr))
        for b in folds:
            m = gtr == b
            predm[m] = fit_mlp(Ftr[~m], ytr[~m], [Ftr[m]], epochs=150, hidden=64)[0][0]
        # full-train val
        vridge = fit_ridge(Ftr, ytr, [Fva], best[1])[0]
        vmlp, _ = fit_mlp(Ftr, ytr, [Fva], epochs=150, hidden=64)
        print(f"{mode:8s} d={Ftr.shape[1]:3d} | LOO ridge a={best[1]:<6} rmse={best[0]:7.1f} | "
              f"LOO mlp rmse={rmse(predm,ytr):7.1f} | val ridge={rmse(vridge,yva):7.1f} val mlp={rmse(vmlp[0],yva):7.1f}")
