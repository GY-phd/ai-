# -*- coding: utf-8 -*-
"""Experiment 6: focused comparison - 4 configs, per-bearing LOO + validation."""
import json
import time
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


def tf(x, mode):
    return x if mode == "raw" else np.sign(x) * np.log1p(np.abs(x))


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

    def step(self, xb, yb, lr):
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
            p[i] -= lr * (self.m[i] / (1 - .9 ** self.t)) / (np.sqrt(self.v[i] / (1 - .999 ** self.t)) + 1e-8)


def fit(X, y, Xv, h=64, epochs=100, lr=3e-3, bs=128, seed=42):
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
            net.step(Z[i], yt[i], cur)
    return np.maximum(0, net.fwd(np.clip((Xv - mu) / sd, -10, 10))[1][:, 0] * ys + ym)


if __name__ == "__main__":
    xtr, ytr, idtr = load(BASE + r"\train.jsonl")
    xva, yva, idva = load(BASE + r"\validation.jsonl")
    gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
    folds = sorted(set(gtr))
    short = {"xjtu_Bearing1_1", "xjtu_Bearing1_2", "xjtu_Bearing1_3", "xjtu_Bearing2_2", "xjtu_Bearing3_3"}
    t0 = time.time()
    for mode in ("raw", "slog"):
        X, Xv = tf(xtr, mode), tf(xva, mode)
        for h in (64, 128):
            v = fit(X, ytr, Xv, h=h)
            per = {}
            pred = np.zeros(len(ytr))
            for b in folds:
                m = gtr == b
                pred[m] = fit(X[~m], ytr[~m], X[m], h=h)
                per[b] = (m.sum(), rmse(pred[m], ytr[m]))
            shorts = [per[b][1] for b in per if b in short]
            print(f"{mode:5s} h={h:3d} | val rmse={rmse(v,yva):7.1f} mae={np.mean(np.abs(v-yva)):7.1f} "
                  f"| LOO pooled={rmse(pred,ytr):7.1f} | short-bearing mean rmse={np.mean(shorts):7.1f} "
                  f"| params={X.shape[1]*h+h+h+1}  ({time.time()-t0:.0f}s)")
            print("     per-bearing: " + " ".join(f"{b.split('_',1)[1]}:{per[b][1]:.0f}" for b in folds))
