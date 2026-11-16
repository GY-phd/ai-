"""查看影子评估的拒绝原因分布。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
case = sys.argv[1] if len(sys.argv) > 1 else "S1_baseline"
p = ROOT / f"artifacts/scen_{case}/exp/experiments.jsonl"
if not p.exists():
    print(f"not found: {p}")
    raise SystemExit(1)

recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
ret = [r for r in recs if r.get("stage") == "retrain"]
print(f"case={case} retrain rounds={len(ret)}")

reasons = Counter()
for r in ret:
    for c in r["config"].get("candidates", []):
        sh = c["shadow"]
        reasons[sh.get("reason", "?")] += 1

print("\nshadow rejection reasons:")
for k, v in reasons.most_common():
    print(f"  {v:3d}  {k}")

print("\ndetail (last 8 rounds):")
for r in ret[-8:]:
    for c in r["config"].get("candidates", []):
        sh = c["shadow"]
        print(
            f"  {sh.get('approved')} delta={sh.get('delta_score'):.5f} "
            f"p={sh.get('p_value'):.4f} n={sh.get('n_samples')} "
            f"base={sh.get('baseline_score'):.4f} cand={sh.get('candidate_score'):.4f} "
            f"| {sh.get('reason', '')[:70]}"
        )
    print("  ---")
