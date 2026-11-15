# -*- coding: utf-8 -*-
"""Experiment 8 (fast): validation-only comparison of feature maps and widths."""
import numpy as np
from exp7 import load, rmse, eng44, eng34, fit, BASE

xtr, ytr, idtr = load(BASE + r"\train.jsonl")
xva, yva, idva = load(BASE + r"\validation.jsonl")
gtr = np.array([i.rsplit("_s", 1)[0] for i in idtr])
gva = np.array([i.rsplit("_s", 1)[0] for i in idva])

print(f"{'map':7s}{'h':>4s}{'wd':>6s}{'ep':>5s} {'val_rmse':>9s}{'val_mae':>9s}   per-bearing")
for name, fn in (("slog", lambda a: np.sign(a) * np.log1p(np.abs(a))), ("eng34", eng34), ("eng44", eng44)):
    X, Xv = fn(xtr), fn(xva)
    for h in (32, 64):
        for wd in (0.0, 0.01):
            v = fit(X, ytr, Xv, h=h, wd=wd, epochs=150)
            per = " ".join(f"{b.split('_',1)[1]}:{rmse(v[gva==b], yva[gva==b]):.0f}" for b in sorted(set(gva)))
            print(f"{name:7s}{h:4d}{wd:6.2f}{150:5d} {rmse(v,yva):9.1f}{np.mean(np.abs(v-yva)):9.1f}   {per}")
