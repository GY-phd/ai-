# -*- coding: utf-8 -*-
"""Experiment 7: MLP on engineered (scale-free + log) features, and a 34->34 bijective variant."""
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


def eng44(x):
    """44 engineered dims (log magnitudes + dimensionless ratios)."""
    cols = []
    for c in range(2):
        g = x[:, c * 17:(c + 1) * 17]
        rms, peak, ptp, mabs, kurt, skew, crest, shape, impulse, margin, sc, srms, skurt, dfreq, b1, b2, b3 = g.T
        tot = np.maximum(b1 + b2 + b3, 1e-12)
        cols += [np.log(np.maximum(np.abs(v), 1e-9)) for v in (rms, peak, ptp, mabs, sc, srms, skurt, b1, b2, b3)]
        cols += [kurt, skew, crest, shape, impulse, margin, b1 / tot, b2 / tot, b3 / tot,
                 sc / 12800.0, dfreq / 12800.0, np.log(np.maximum(skurt, 1e-9))]
    return np.stack(cols, axis=1)


def eng34(x):
    """34->34 reparameterisation: same width, but ratios/logs instead of raw levels."""
    out = np.empty_like(x)
    for c in range(2):
        g = x[:, c * 17:(c + 1) * 17]
        rms, peak, ptp, mabs, kurt, skew, crest, shape, impulse, margin, sc, srms, skurt, dfreq, b1, b2, b3 = g.T
        tot = np.maximum(b1 + b2 + b3, 1e-12)
        out[:, c * 17 + 0] = np.log(np.maximum(rms, 1e-9))
        out[:, c * 17 + 1] = np.log(np.maximum(peak, 1e-9))
        out[:, c * 17 + 2] = np.log(np.maximum(ptp, 1e-9))
        out[:, c * 17 + 3] = np.log(np.maximum(mabs, 1e-9))
        out[:, c * 17 + 4] = kurt
        out[:, c * 17 + 5] = skew
        out[:, c * 17 + 6] = crest
        out[:, c * 17 + 7] = shape
        out[:, c * 17 + 8] = impulse
        out[:, c * 17 + 9] = margin
        out[:, c * 17 + 10] = sc / 12800.0
        out[:, c * 17 + 11] = np.log(np.maximum(srms, 1e-9))
        out[:, c * 17 + 12] = np.log(np.maximum(skurt, 1e-9))
        out[:, c * 17 + 13] = dfreq / 12800.0
        out[:, c * 17 + 14] = b1 / tot
        out[:, c * 17 + 15] = b2 / tot
        out[:, c * 17 + 16] = b3 / tot
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

    def step(self, xb, yb, lr, wd=0.0):
        h, o = self.fwd(xb)
        d = (2 * (o[:, 0] - yb) / len(xb))[:, None]
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


def fit(X, y, Xv, h=64, epochs=150, lr=3e-3, bs=128, wd=0.0, seed=42):
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
            net.step(Z[i], yt[i], cur, wd)
    return np.maximum(0, net.fwd(np.clip((Xv - mu) / sd, -10, 10))[1][:, 0] * ys + ym)


if __name__ == "__main__":
    xtr, ytr, idtr = load(BASE + r"\train.jsonl")
    xva, yva, idva = load(BASE + r"\validation.jsonl")
    gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
    folds = sorted(set(gtr))
    for name, fn in (("eng44", eng44), ("eng34", eng34)):
        X, Xv = fn(xtr), fn(xva)
        for h in (32, 64):
            for wd in (0.0, 0.01):
                v = fit(X, ytr, Xv, h=h, wd=wd)
                pred = np.zeros(len(ytr))
                for b in folds:
                    m = gtr == b
                    pred[m] = fit(X[~m], ytr[~m], X[m], h=h, wd=wd)
                print(f"{name} d={X.shape[1]:3d} h={h:2d} wd={wd:.2f} | val rmse={rmse(v,yva):7.1f} mae={np.mean(np.abs(v-yva)):7.1f} "
                      f"| LOO={rmse(pred,ytr):7.1f}")
