"""分析官方数据包: 轴承划分、RUL 分布、基线性能。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"d:\GY\deepseek\cpu-b-rul-20261110-v1")
sys.path.insert(0, str(ROOT))


def load(path):
    ids, metas, answers = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(row["id"])
        metas.append(row.get("meta", {}))
        answers.append(row["answer"])
    return ids, metas, np.asarray(answers, dtype=np.float64)


def report(name, path):
    ids, metas, y = load(path)
    by_bearing = defaultdict(list)
    for m, v in zip(metas, y):
        by_bearing[m.get("bearing", "?")].append((m.get("snapshot", -1), v))
    print(f"\n=== {name} ===")
    print(f"samples: {len(ids)}  bearings: {len(by_bearing)}")
    print(
        f"RUL: min={y.min():.0f} max={y.max():.0f} mean={y.mean():.1f} median={np.median(y):.0f}"
    )
    print(f"{'bearing':<16}{'n_snap':>8}{'maxRUL':>10}{'firstRUL':>11}{'contiguous':>12}")
    for b in sorted(by_bearing):
        rows = sorted(by_bearing[b])
        snaps = [s for s, _ in rows]
        print(
            f"{b:<16}{len(rows):>8}{rows[-1][1]:>10.0f}{rows[0][1]:>11.0f}"
            f"{str(snaps == list(range(1, len(snaps)+1))):>12}"
        )
    return by_bearing, y


tr_bear, tr_y = report("TRAIN", ROOT / "public/train.jsonl")
va_bear, va_y = report("VALIDATION", ROOT / "public/validation.jsonl")

print("\n=== bearing isolation ===")
print(f"train bearings : {sorted(tr_bear)}")
print(f"valid bearings : {sorted(va_bear)}")
print(f"overlap        : {sorted(set(tr_bear) & set(va_bear))}  (must be empty)")

print("\n=== lifecycle drift signal ===")
from baselines.data import read_dataset  # noqa: E402

train = read_dataset(ROOT / "public/train.jsonl", "regression", [])
valid = read_dataset(
    ROOT / "public/validation.jsonl", "regression", [], train["paths"]
)
print(f"feature dim: {train['x'].shape[1]}  train: {train['x'].shape}  valid: {valid['x'].shape}")

paths = train["paths"]
print("\nfirst 5 feature paths:")
for p in paths[:5]:
    print(f"  {p}")

kurt_idx = [i for i, p in enumerate(paths) if "kurtosis" in p]
bands = [(0, 30), (30, 60), (60, 120), (120, 240), (240, 10**9)]
print(f"\n{'RUL range':<16}{'n':>8}{'kurt_horiz':>14}{'kurt_vert':>14}")
for lo, hi in bands:
    m = (tr_y >= lo) & (tr_y < hi)
    if m.sum() == 0:
        continue
    kh = train["x"][m, kurt_idx[0]].mean() if len(kurt_idx) > 0 else float("nan")
    kv = train["x"][m, kurt_idx[1]].mean() if len(kurt_idx) > 1 else float("nan")
    print(f"{f'[{lo},{hi})':<16}{m.sum():>8}{kh:>14.3f}{kv:>14.3f}")

print("\n=== baseline checkpoint on validation ===")
from baselines.model import Model  # noqa: E402

model = Model.load(ROOT / "checkpoint.npz")
pred = model.predict(valid["x"])
err = pred - valid["y"]
print(f"MAE  = {np.mean(np.abs(err)):.3f} min")
print(f"RMSE = {np.sqrt(np.mean(err**2)):.3f} min")
print(f"baseline RUL mean={valid['y'].mean():.1f}  pred mean={pred.mean():.1f}")
