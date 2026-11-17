"""RUL 赛题闭环的严格性审查 —— 检查是否存在评估泄漏。

关键问题:
  1. 训练数据是否包含了流中样本自身的标签? (自标签泄漏)
  2. 影子评估集是否被用于训练? (影子泄漏)
  3. 静态基线的对比是否公平? (同口径)
  4. RUL 标签在流中是否可提前获知? (时序泄漏)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evo.config import load_config
from evo.harness import Harness

cfg = load_config(str(ROOT / "configs/rul_bearing.yaml"))
cfg.set("paths.artifacts_root", "artifacts/audit_rul")
cfg.set("paths.audit_log", "artifacts/audit_rul/audit/audit.jsonl")
cfg.set("paths.shadow_root", "artifacts/audit_rul/shadow")
cfg.set("experiment.local_dir", "artifacts/audit_rul/exp")

h = Harness(cfg, use_synthetic=False)
h.prepare_data()

# ---- 记录拆分后的信息 ----
train_bearings = set(h.train_ds.meta["bearing"].unique())
stream_bearings = set(h.stream.meta["bearing"].unique())
valid_bearings = set(h.valid_ds.meta["bearing"].unique())

print("=== 1. 数据拆分 ===")
print(f"train bearings ({len(train_bearings)}): {sorted(train_bearings)}")
print(f"stream bearings ({len(stream_bearings)}): {sorted(stream_bearings)}")
print(f"valid bearings ({len(valid_bearings)}): {sorted(valid_bearings)}")
print(f"train ∩ stream = {sorted(train_bearings & stream_bearings)}  (必须为空)")
print(f"train ∩ valid  = {sorted(train_bearings & valid_bearings)}  (必须为空)")
print(f"stream n={len(h.stream)}  train n={len(h.train_ds)}")

# 流的结构: 留出轴承的生命周期
print("\n=== 2. 流结构 (轴承生命周期) ===")
for b, grp in h.stream.meta.groupby("bearing"):
    idx = grp.index.to_numpy()
    rul = h.stream.y[idx]
    print(
        f"  {b}: n={len(idx)} snapshot {grp['snapshot'].min()}..{grp['snapshot'].max()} "
        f"RUL {rul.min():.0f}..{rul.max():.0f} (均值 {rul.mean():.0f})"
    )

# ---- 跑闭环并记录每轮的训练窗口 ----
print("\n=== 3. 运行闭环, 记录每轮训练数据来源 ===")
summary = h.run()

print(f"\niterations={summary.iterations}  approved={summary.updates_approved}")
print(f"tail: baseline={summary.baseline_tail_score:.2f}  adaptive={summary.tail_score:.2f}")
print(f"RMSE tail: baseline={-summary.baseline_tail_score:.2f} min  adaptive={-summary.tail_score:.2f} min")

print("\n=== 4. 逐轮检查: 影子集是否进入训练 ===")
recs = [r for r in h.recorder.local.load() if r.get("stage") == "retrain"]
print(f"重训轮次: {len(recs)}")

# 检查最后一个窗口: 训练数据是否覆盖了影子集
last = h.iterations[-1] if h.iterations else None
if last is not None:
    print(f"最后窗口: {last.window}")
    shadow = h._shadow_set(last.window)
    print(f"影子集: [{shadow.meta['timestamp'].min():.0f}, {shadow.meta['timestamp'].max():.0f}] n={len(shadow)}")
    print(f"影子集快照号范围: {shadow.meta['snapshot'].min()}..{shadow.meta['snapshot'].max()}")

print("\n=== 5. 静态基线对比是否同口径 ===")
print(f"static_baseline 是否独立冻结: {h.static_baseline is not h.current_model}")
print(f"static version: {h.static_baseline.version}")
print(f"final  version: {h.current_model.version}")
print(f"在线打分次数 (自适应/基线): {len(h.online_scores)}/{len(h.online_baseline_scores)}")
if h.online_scores and h.online_baseline_scores:
    a = np.array(h.online_scores)
    b = np.array(h.online_baseline_scores)
    print(f"逐窗口对比 (前 8):")
    for i in range(min(8, len(a))):
        print(f"  win{i}: adaptive={-a[i]:8.1f}  baseline={-b[i]:8.1f}  diff={-(a[i]-b[i]):+8.1f}")
    print(f"  改善窗口数: {int(np.sum(a > b))}/{len(a)}")
    print(f"  总体: 自适应 RMSE {-a.mean():.1f} vs 基线 {-b.mean():.1f} min")
