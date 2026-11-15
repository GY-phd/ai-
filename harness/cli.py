"""Command line entry point for the minimal harness."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

from .baseline import default_baseline_root
from .config import load_config
from .engine import HarnessEngine, discover_batches
from .state import STATE_FILE, champion_summary, load_state


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli",
        description="Minimal autonomous training harness for the CPU-C RUL task.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="consume a stream of batches and decide per batch")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--stream", help="directory containing batch-*/validation.jsonl, sorted by name")
    source.add_argument("--batch", action="append", metavar="DIR",
                        help="explicit batch directory; may be repeated and is processed in order")
    run.add_argument("--workdir", required=True, help="persistent state, models, decisions and journal")
    run.add_argument("--baseline-root", default=str(default_baseline_root()),
                     help="official CPU-C baseline package (never modified)")
    run.add_argument("--config", default=str(DEFAULT_CONFIG), help="harness policy JSON")
    run.add_argument("--max-batches", type=int, help="process at most this many batches")
    run.add_argument("--dry-run", action="store_true",
                     help="detect, diagnose and plan experiments without training or advancing state")
    run.add_argument("--force", action="store_true", help="reprocess batches already present in state")
    run.add_argument("--swanlab", choices=("off", "offline", "cloud", "required"),
                     help="override logging.swanlab for this run")
    run.add_argument("--json", action="store_true", help="print a machine-readable run summary")
    run.add_argument("--quiet", action="store_true", help="suppress progress messages on stderr")

    status = subparsers.add_parser("status", help="show champion lineage and batch ledger")
    status.add_argument("--workdir", required=True)
    status.add_argument("--json", action="store_true")
    return parser


def _load_status(workdir: Path) -> dict:
    state_path = Path(workdir) / STATE_FILE
    if not state_path.is_file():
        raise ValueError(f"no harness state at {state_path}; run a batch stream first")
    state = load_state(state_path)
    return {
        "format": "harness.status.v1",
        "workdir": str(Path(workdir).resolve()),
        "champion": champion_summary(state["champion"]),
        "previous_champion": champion_summary(state["previous_champion"])
        if state.get("previous_champion") else None,
        "counters": state["counters"],
        "batches": state["batches"],
        "guard": state["data"].get("guard"),
        "replay": state["data"].get("replay"),
    }


def _explicit_batches(paths: list[str]) -> list[Path]:
    batches = []
    seen = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"batch is not a regular directory: {raw}")
        validation = path / "validation.jsonl"
        if validation.is_symlink() or not validation.is_file():
            raise ValueError(f"batch has no regular validation.jsonl: {raw}")
        if path.name in seen:
            raise ValueError(f"duplicate batch directory name: {path.name}")
        seen.add(path.name)
        batches.append(path)
    return batches


def _print_status(status: dict) -> None:
    champion = status["champion"]
    metrics = champion.get("metrics") or {}
    batch_rmse = (metrics.get("batch") or {}).get("rmse")
    guard_rmse = (metrics.get("guard") or {}).get("rmse")
    print(f"workdir: {status['workdir']}")
    print(f"champion: {champion.get('stage_id')} (generation {champion.get('generation')}, "
          f"static={champion.get('is_static')})")
    print(f"  checkpoint: {champion.get('checkpoint')}")
    print(f"  reference RMSE: batch={batch_rmse} guard={guard_rmse}")
    counters = status["counters"]
    print("counters: " + " ".join(f"{key}={value}" for key, value in counters.items()))
    print(f"batches: {len(status.get('batches') or [])}")
    for record in status.get("batches") or []:
        print(f"  {record.get('batch_id')}: {record.get('decision')} "
              f"drift={record.get('drift_severity')} experiments={record.get('experiments')} "
              f"-> {record.get('decision_file')}")
    previous = status.get("previous_champion")
    if previous:
        print(f"previous champion: {previous.get('stage_id')} "
              f"(generation {previous.get('generation')})")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            status = _load_status(Path(args.workdir))
            if args.json:
                print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
            else:
                _print_status(status)
            return 0

        if args.max_batches is not None and args.max_batches < 1:
            parser.error("--max-batches must be at least 1")
        config = load_config(args.config)
        if args.swanlab:
            config = replace(config, logging=replace(config.logging, swanlab=args.swanlab))
        batches = discover_batches(args.stream) if args.stream else _explicit_batches(args.batch)
        engine = HarnessEngine(
            workdir=args.workdir, config=config, baseline_root=args.baseline_root,
            dry_run=args.dry_run, force=args.force, verbose=not args.quiet,
        )
        decisions = engine.run(batches, max_batches=args.max_batches)
        status = engine.status()
        if args.json:
            print(json.dumps({"format": "harness.run-summary.v1", "dry_run": args.dry_run,
                              "decisions": decisions, "status": status},
                             ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(f"workdir: {engine.workdir}")
            for record in decisions:
                print(f"{record['batch']['id']}: {record['decision'].upper()} "
                      f"-> {record['decision_file']}")
                print(f"  {record['reason']}")
            if not decisions:
                print("no new batches processed (all already present in state; use --force to rerun)")
            champion = status["champion"]
            print(f"champion: {champion.get('stage_id')} (generation {champion.get('generation')})")
            print("counters: " + " ".join(f"{key}={value}" for key, value in status["counters"].items()))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"harness error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if argv is None:
            return 1
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
