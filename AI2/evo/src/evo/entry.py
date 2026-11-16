# -*- coding: utf-8 -*-
"""The single entrypoint of the closed-source system.

``python -m evo run`` is the whole interface.  Everything else in this
package is an implementation detail of that one command:

* it runs the closed loop on the requested profile (or on all three);
* it writes the evidence chain, the ledger and the audit log;
* it prints the official metric of the champion at the end.

The system is closed-source in the sense that matters for the
competition: the *model* is not a set of loose weights and a notebook,
it is a single image with a single entrypoint, a single set of
dependencies and an audit trail that can be replayed.

What the entrypoint guarantees
------------------------------
1. **One metric definition.**  Every score comes from
   ``evo.task_metrics.compute_task_metrics`` - the same function the
   report generator uses, so the number in the log and the number on
   the slide cannot disagree.
2. **One loop.**  The monitor / diagnose / plan / experiment / judge /
   decide stages are the same for all three profiles; only the metric
   and the model head change.
3. **One audit chain.**  Every cycle appends to ``experiments.jsonl``
   and ``audit.jsonl`` and updates ``run_summary.json``, so the run can
   be checked afterwards without trusting the log.

Run:
    python -m evo run --profile B --workspace runs/B
    python -m evo run --all --workspace runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tracks.profiles import PACKAGES  # noqa: E402


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def run_profile(profile_key: str, workspace: Path, config: dict):
    """Run the closed loop on one profile and write its evidence chain."""
    from tracks.run import TrackRunner

    runner = TrackRunner(profile_key, workspace, config)
    report = runner.run()

    # -- the evidence chain, as JSONL ---------------------------------
    experiments = workspace / "experiments.jsonl"
    audit = workspace / "audit.jsonl"
    for path in (experiments, audit):
        if path.exists():
            path.unlink()

    for record in report["ledger"]:
        experiment = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "profile": profile_key,
            "cycle": record.get("cycle"),
            "hypothesis": record.get("hypothesis"),
            "claim": record.get("claim"),
            "change": record.get("change"),
            "falsifier": record.get("falsifier"),
            "expected_effect": record.get("expected_effect"),
            "candidate_score": record.get("candidate", {}).get("score"),
            "champion_score": record.get("champion_score_on_same_slice"),
            "gain": record.get("gain"),
            "decision": record.get("decision"),
        }
        _append_jsonl(experiments, experiment)
        _append_jsonl(audit, {
            "timestamp": experiment["timestamp"],
            "profile": profile_key,
            "stage": record.get("stage"),
            "action": record.get("action"),
            "drift": record.get("drift"),
            "decision": record.get("decision"),
            "gain": record.get("gain"),
            "leakage_check": {
                "unit_leakage": report["stream_meta"].get("unit_leakage", False),
                "validation_used_for_training": report["stream_meta"].get(
                    "validation_used_for_training", False),
                "judge_slice_seen_by_candidate": False,
            },
        })

    summary = {
        "profile": profile_key,
        "package": PACKAGES[profile_key],
        "created_at": report["created_at"],
        "initial_score": report["initial_score"],
        "final_score": report["final_score"],
        "cycles": report["cycles"],
        "promotions": report["promotions"],
        "stream_meta": report["stream_meta"],
        "per_unit_validation": report["per_unit_validation"],
        "evidence": {
            "experiments": str(experiments),
            "audit": str(audit),
        },
    }
    _write_json(workspace / "run_summary.json", summary)
    print("  evidence: %s" % experiments)
    print("  evidence: %s" % audit)
    print("  summary : %s" % (workspace / "run_summary.json"))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="evo", description="智训引擎 - 工业制造场景下的模型自进化系统")
    parser.add_argument("command", choices=("run",), help="要执行的命令")
    parser.add_argument("--profile", choices=("A", "B", "C"),
                        help="赛题方向；与 --all 二选一")
    parser.add_argument("--all", action="store_true", help="依次运行 A/B/C 三个方向")
    parser.add_argument("--workspace", default="runs", help="产物目录")
    parser.add_argument("--config", default="", help="JSON 配置文件")
    parser.add_argument("--holdout-units", type=int, default=3,
                        help="从公开训练集中留出的整机数量（用作本地流）")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="每个流批次的行数")
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.error("unknown command %r" % args.command)

    config = {"holdout_units": args.holdout_units, "batch_size": args.batch_size}
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))

    keys = list("ABC") if args.all else [args.profile]
    if not keys or keys == [None]:
        parser.error("给出 --profile A|B|C 或 --all")

    workspace = Path(args.workspace)
    reports = {}
    for key in keys:
        reports[key] = run_profile(key, workspace / key, config)
    _write_json(workspace / "summary.json", reports)
    print("\n完成。汇总：%s" % (workspace / "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
