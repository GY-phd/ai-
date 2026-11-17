"""检查 RUL 场景下策略层的探索是否有效。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "artifacts/rulcmp_E3_full_s20261110/exp/experiments.jsonl"
recs = [
    json.loads(line)
    for line in p.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ret = [r for r in recs if r.get("stage") == "retrain"]
print(f"retrain rounds: {len(ret)}")

nc = {}
for r in ret:
    n = len(r["config"].get("candidates", []))
    nc[n] = nc.get(n, 0) + 1
print(f"candidates per round: {nc}")

champ_wins = chal_wins = 0
for i, r in enumerate(ret):
    parts = []
    for c in r["config"].get("candidates", []):
        sh = c["shadow"]
        mark = "*" if c["arm"] == r["config"].get("chosen_arm") else " "
        pol = c["policy"]
        parts.append(
            f"[{mark}lr={pol['lr']},lv={pol['num_leaves']},"
            f"win={pol['retrain_window']},mg={pol['min_gain']} "
            f"sc={sh['candidate_score']:.0f}]"
        )
    chosen = r["config"].get("chosen_arm")
    cands = r["config"].get("candidates", [])
    if len(cands) > 1:
        first = cands[0]
        if chosen == first["arm"]:
            champ_wins += 1
        else:
            chal_wins += 1
    if i < 10:
        print(f"  r{i + 1:02d} " + " ".join(parts))

print(f"\nchampion wins: {champ_wins}  challenger wins: {chal_wins}")

# 冠军参数轨迹
print("\nchampion policy over time:")
for r in ret:
    chosen = r["config"].get("chosen_arm")
    cands = r["config"].get("candidates", [])
    if not cands:
        continue
    pol = next((c["policy"] for c in cands if c["arm"] == chosen), cands[0]["policy"])
    print(
        f"  psi_trig={pol['psi_trigger']} min_gain={pol['min_gain']} "
        f"win={pol['retrain_window']} lr={pol['lr']} lv={pol['num_leaves']}"
    )
