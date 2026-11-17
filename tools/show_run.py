# -*- coding: utf-8 -*-
"""Render a harness run as a human-readable decision table (for reports / review)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="runs/harness-b")
    args = parser.parse_args()
    root = Path(args.workspace)
    report = json.loads((root / "run_report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"run      : {report['run_name']}")
    print(f"planner  : {report['planner']['planner']} (model={report['llm_model_id']}, "
          f"tokens={report['llm_tokens_used']})")
    print(f"baseline : RMSE {report['baseline']['rmse']:.2f} / MAE {report['baseline']['mae']:.2f}")
    print(f"final    : RMSE {report['final']['rmse']:.2f} / MAE {report['final']['mae']:.2f} "
          f"(params {report['final']['parameter_count']})")
    print(f"gain     : {report['improvement_pct']:.2f}%   cycles={report['cycles']}   "
          f"experiments={report['budget']['experiments_used']}/{report['budget']['experiments_used'] + report['budget']['experiments_left']}   "
          f"train={report['budget']['train_seconds_used']:.0f}s")
    print()
    header = f"{'#':>3} {'batch':>5} {'diagnosis':<11} {'action':<18} {'decision':<8} {'cand':>7} {'champ':>7} {'cost':>6}  reason"
    print(header)
    print("-" * len(header))
    champ = None
    for r in rows:
        if r.get("stage") == "bootstrap":
            champ = r["metrics_after"]["rmse"]
            continue
        if r.get("stage") != "decide":
            continue
        cand = (r.get("candidate_metrics") or {}).get("rmse")
        if r.get("decision") == "promote":
            champ = cand
        ratio = f"{cand:7.1f}" if cand is not None else "      -"
        print(f"{r['cycle']:>3} {r.get('batch', '-'):>5} {r.get('diagnosis', {}).get('drift_type', '-'):<11} "
              f"{r['action']:<18} {r['decision']:<8} {ratio} {champ if champ is not None else float('nan'):7.1f} "
              f"{r['cost_seconds']:6.0f}  {(r.get('gate') or {}).get('reason', '')[:60]}")

    print("\nchampion chain:")
    for entry in report["champion_history"]:
        print(f"  cycle {entry['cycle']}  {entry['source']:<18} rmse={entry['rmse']:.2f}  {entry['checkpoint_sha256'][:22]}...")
    print("\nper bearing (final):")
    for name, row in report["final"]["per_bearing"].items():
        print(f"  {name:<22} rows={row['rows']:>5}  rmse={row['rmse']:8.1f}  mae={row['mae']:8.1f}")


if __name__ == "__main__":
    main()
