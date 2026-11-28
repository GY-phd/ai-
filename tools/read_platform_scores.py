# -*- coding: utf-8 -*-
"""Record the two formal platform evaluations of 2026-11-25.

Read the platform's own report API and write the numbers into a table so
that the documents can cite one source of truth.

    python tools/read_platform_scores.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "平台正式评测成绩_20261125.md"

# The objective-score components and their caps, from the platform's rules
# page ("客观评分 35 分：平均性能提升 15 分、最差阶段与恢复速度 10 分、
# 稳定性 5 分、资源效率 5 分").
CAPS = {
    "mean_gain": (15, "平均性能提升"),
    "worst_stage": (5, "最差阶段表现"),
    "recovery": (5, "恢复速度"),
    "retention": (5, "稳定性"),
    "resource_efficiency": (5, "资源效率"),
}

GATE_LABELS = {
    "autonomous_completion_on_time": "无人干预按时完成",
    "three_traceable_decision_cycles": "至少 3 次完整决策循环",
    "not_below_static_baseline": "效果不低于平台基线",
    "experiment_records_complete": "实验记录完整",
    "complete_materials_and_reproducibility": "材料齐全且代码可复现",
}


def main(argv=None):
    """Print the score table from a saved report JSON.

    The reports are read from the platform by the agent (the session
    token lives in the browser), so this script works on a dump:

        python tools/read_platform_scores.py path/to/report.json ...
    """
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        p = pathlib.Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        obj = data.get("objective_result") or {}
        comps = obj.get("objective_components") or {}
        gates = obj.get("gates") or {}
        print("== %s" % p.name)
        print("   objective_score = %.4f / 35" % (obj.get("objective_score") or 0))
        for key, (cap, label) in CAPS.items():
            v = comps.get(key)
            if v is None:
                continue
            print("   %-12s %-8s %8.4f / %-3d  (%.1f%%)"
                  % (key, label, v, cap, v / cap * 100))
        for key, label in GATE_LABELS.items():
            if key in gates:
                print("   gate %-38s %s" % (label, "PASS" if gates[key] else "FAIL"))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
