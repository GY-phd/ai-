# -*- coding: utf-8 -*-
"""Experiment 9: does bearing-balanced weighting or prior shrinkage generalise better?

Row-weighted training matches the competition metric (RMSE over rows), but the
row counts per bearing differ by 20x, so the loss is dominated by two long
bearings.  Here we compare three weighting schemes and a shrinkage sweep on the
public validation slice, and report per-bearing errors because the fleet mean
hides exactly the failure we care about.
"""
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
        v = [float(f[ch][k]) for ch in ("horizontal", "vertical") for k in sorted(f[ch])]
        xs.append(v); ys.append(float(r["answer"])); ids.append(r["id"])
    return np.array(xs), np.array(ys), ids


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def eng34(x):
    out = np.empty_like(x)
    names = ["band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k", "crest_factor",
             "dominant_freq", "impulse_factor", "kurtosis", "margin_factor", "mean_abs", "peak",
             "peak_to_peak", "rms", "shape_factor", "skewness", "spectral_centroid",
             "spectral_kurtosis", "spectral_rms"]
    idx = {n: i for i, n in enumerate(names)}
    for c in range(2):
        g = x[:, c * 17:(c + 1) * 17]
        f = {n: g[:, i] for n, i in idx.items()}
        tot = np.maximum(f["band1_energy_0_2k"] + f["band2_energy_2k_6k"] + f["band3_energy_6k_12k"], 1e-12)
        o = out[:, c * 17:(c + 1) * 17]
        for n in ("rms", "peak", "peak_to_peak", "mean_abs", "spectral_rms", "spectral_kurtosis"):
            o[:, idx[n]] = np.log(np.maximum(f[n], 1e-9))
        for n in ("kurtosis", "skewness", "crest_factor", "shape_factor", "impulse_factor", "margin_factor"):
            o[:, idx[n]] = f[n]
        o[:, idx["spectral_centroid"]] = f["spectral_centroid"] / 12800.0
        o[:, idx["dominant_freq"]] = f["dominant_freq"] / 12800.0
        o[:, idx["band1_energy_0_2k"]] = f["band1_energy_0_2k"] / tot
        o[:, idx["band2_energy_2k_6k"]] = f["band2_energy_2k_6k"] / tot
        o[:, idx["band3_energy_6k_12k"]] = f["band3_energy_6k_12k"] / tot
    return out


class MLP:
    def __init__(self, d, h, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1 / d), (d, h)); self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(1 / h), (h, 1)); self.b2 = np.zeros(1)
        self.m = [np.zeros_like(a) for a in (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(a) for a in (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0

    def fwd(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def step(self, xb, yb, w, lr, wd=0.0):
        h, o = self.fwd(xb)
        err = o[:, 0] - yb
        d = (2.0 * w * err / max(float(w.sum()), 1e-12))[:, None]
        gW2 = h.T @ d; gb2 = d.sum(0)
        dh = (d @ self.W2.T) * (1 - h ** 2)
        g = [xb.T @ dh, dh.sum(0), gW2, gb2]
        n = np.sqrt(sum(float((x * x).sum()) for x in g))
        s = min(1.0, 5.0 / max(n, 1e-12))
        self.t += 1
        p = [self.W1, self.b1, self.W2, self.b2]
        for i in range(4):
            g[i] *= s
            self.m[i] = .9 * self.m[i] + .1 * g[i]
            self.v[i] = .999 * self.v[i] + .001 * g[i] ** 2
            upd = (self.m[i] / (1 - .9 ** self.t)) / (np.sqrt(self.v[i] / (1 - .999 ** self.t)) + 1e-8)
            p[i] -= lr * (upd + wd * p[i])


def fit(X, y, Xv, w, h=96, epochs=400, lr=3e-3, bs=128, wd=0.01, seed=42):
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1
    Z = np.clip((X - mu) / sd, -10, 10)
    ym, ys = y.mean(), max(y.std(), 1e-9)
    yt = (y - ym) / ys
    net = MLP(Z.shape[1], h, seed)
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Z))
        cur = lr * 0.5 * (1 + np.cos(np.pi * (ep - 1) / epochs))
        for s in range(0, len(order), bs):
            i = order[s:s + bs]
            net.step(Z[i], yt[i], w[i], cur, wd)
    return np.maximum(0, net.fwd(np.clip((Xv - mu) / sd, -10, 10))[1][:, 0] * ys + ym)


if __name__ == "__main__":
    xtr, ytr, idtr = load(BASE + r"\train.jsonl")
    xva, yva, idva = load(BASE + r"\validation.jsonl")
    Ftr, Fva = eng34(xtr), eng34(xva)
    gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
    gva = np.array([i.rsplit("_s", 1)[0] for i in idva])
    counts = {g: int((gtr == g).sum()) for g in set(gtr)}
    schemes = {
        "uniform": np.ones(len(ytr)),
        "bearing_balanced": np.array([1.0 / counts[g] for g in gtr]),
        "sqrt_balanced": np.array([1.0 / np.sqrt(counts[g]) for g in gtr]),
    }
    for name, w in schemes.items():
        w = w / w.mean()
        pred = fit(Ftr, ytr, Fva, w)
        per = " ".join(f"{b.split('_',1)[1]}:{rmse(pred[gva==b], yva[gva==b]):.0f}" for b in sorted(set(gva)))
        print(f"{name:18s} val rmse={rmse(pred,yva):7.1f} mae={np.mean(np.abs(pred-yva)):7.1f}   {per}")
        # shrinkage toward the training prior
        for lam in (0.9, 0.8, 0.7):
            shrunk = lam * pred + (1 - lam) * ytr.mean()
            print(f"   shrink={lam:.1f}      rmse={rmse(shrunk,yva):7.1f}")
