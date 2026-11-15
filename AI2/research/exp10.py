# -*- coding: utf-8 -*-
"""Seed stability of the from-scratch eng34 model (is 426 reproducible?)."""
import numpy as np
from exp9 import load, rmse, eng34, fit

BASE = r"D:\GY\deepseek\cpu-b-rul-20261110-v1\public"

if __name__ == "__main__":
    xtr, ytr, idtr = load(BASE + r"\train.jsonl")
    xva, yva, idva = load(BASE + r"\validation.jsonl")
    Ftr, Fva = eng34(xtr), eng34(xva)
    gva = np.array([i.rsplit("_s", 1)[0] for i in idva])
    w = np.ones(len(ytr))
    for seed in (42, 43, 44):
        for epochs in (400, 600):
            pred = fit(Ftr, ytr, Fva, w, epochs=epochs, seed=seed)
            per = " ".join(f"{b.split('_',1)[1]}:{rmse(pred[gva==b], yva[gva==b]):.0f}" for b in sorted(set(gva)))
            print(f"seed={seed} ep={epochs} val={rmse(pred,yva):7.1f}  {per}")
