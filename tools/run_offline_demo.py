#!/usr/bin/env python3
"""One-command offline closed-loop demo.

Generates a synthetic drift stream, runs the full Harness once, writes an
evidence report and (when SwanLab is installed) records the complete run in
SwanLab offline mode.  The core loop never uses the network or a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.config import config_to_dict, load_config  # noqa: E402
from harness.doctor import collect_environment  # noqa: E402
from harness.engine import HarnessEngine, discover_batches  # noqa: E402
from harness.util import now_utc, unique_path, write_json_atomic  # noqa: E402

DEFAULT_BASELINE = REPO_ROOT / "cpu-c-guide-20261110-v1"
DEFAULT_CONFIG = REPO_ROOT / "harness" / "configs" / "demo.json"


def _fmt(value, digits=3):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _report(*, workdir: Path, stream: Path, report_path: Path, environment: dict,
            decisions: list[dict], status: dict, swanlab_before: dict,
            swanlab_after: dict, wall_seconds: float, error: str | None = None) -> None:
    counters = status.get("counters") or {}
    champion = status.get("champion") or {}
    champion_metrics = champion.get("metrics") or {}
    lines = [
        "# Offline Harness Closed-loop Report",
        "",
        f"- generated_at: {now_utc()}",
        f"- workdir: `{workdir}`",
        f"- stream: `{stream}`",
        f"- harness: offline / CPU-only / no GPU",
        f"- wall_seconds: {wall_seconds:.2f}",
        "",
        "## Environment",
        "",
        f"- Python: {environment['python']}",
        f"- NumPy: {environment['numpy']}",
        f"- CPU: {environment['cpu_count']} cores, {environment['cpu_threads_requested']} requested threads",
        f"- GPU required: {environment['gpu_required']}",
        f"- Network required: {environment['network_required']}",
        f"- Baseline: `{environment['baseline']['root']}`",
        f"- Checkpoint: `{environment['baseline']['checkpoint_sha256']}`",
        f"- SwanLab installed: {environment['swanlab']['installed']} "
        f"(version {environment['swanlab']['version']}, mode {environment['swanlab']['mode']})",
        "",
        "## Decisions",
        "",
        "| batch | decision | drift | experiments | champion RMSE | static RMSE | selected RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in decisions:
        metrics = record.get("metrics") or {}
        champion_batch = (metrics.get("champion_before") or {}).get("batch") or {}
        static_batch = (metrics.get("static") or {}).get("batch") or {}
        selected_batch = ((record.get("selected_model") or {}).get("metrics") or {}).get("batch") or {}
        lines.append(
            f"| {record.get('batch', {}).get('id')} | {record.get('decision')} | "
            f"{(record.get('drift') or {}).get('severity')} | "
            f"{len(record.get('experiments') or [])} | "
            f"{_fmt(champion_batch.get('rmse'))} | {_fmt(static_batch.get('rmse'))} | "
            f"{_fmt(selected_batch.get('rmse'))} |")
    lines += ["", "### Reasons", ""]
    for record in decisions:
        lines.append(f"- **{record.get('batch', {}).get('id')}**: {record.get('reason')}")
    lines += [
        "",
        "## Counters and champion",
        "",
        f"- champion: `{champion.get('stage_id')}` (generation {champion.get('generation')}, "
        f"static={champion.get('is_static')})",
        f"- champion metrics: batch RMSE {_fmt((champion_metrics.get('batch') or {}).get('rmse'))}, "
        f"guard RMSE {_fmt((champion_metrics.get('guard') or {}).get('rmse'))}",
        "- counters: " + ", ".join(f"{key}={value}" for key, value in counters.items()),
        f"- total experiments: {counters.get('attempts', 0)}",
        "",
        "## SwanLab",
        "",
        f"- mode: {swanlab_before.get('mode')}",
        f"- enabled: {swanlab_before.get('enabled')}",
        f"- run_id: {swanlab_before.get('run_id')}",
        f"- run_dir: `{swanlab_before.get('run_dir')}`",
        f"- url: {swanlab_before.get('url')}",
        f"- log_dir: `{swanlab_before.get('log_dir')}`",
        f"- stopped_reason: {swanlab_after.get('disabled_reason')}",
        "",
        "Logged when SwanLab is available: bootstrap baseline metrics, per-batch drift signals, "
        "gate outcomes, champion/static/candidate metrics, per-epoch loss/validation curves, "
        "resource usage, text hypotheses and final summary.",
        "",
        "## Artifacts",
        "",
        f"- state: `{workdir / 'state.json'}`",
        f"- journal: `{workdir / 'journal.jsonl'}`",
        f"- decisions: `{workdir / 'decisions'}`",
        f"- models: `{workdir / 'models'}`",
        f"- effective config: `{workdir / 'effective-config.json'}`",
        "",
    ]
    if error:
        lines += ["## Error", "", f"```\n{error}\n```", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE))
    parser.add_argument("--workdir", default=str(REPO_ROOT / "runs" / "offline-demo"))
    parser.add_argument("--stream", default=str(REPO_ROOT / "runs" / "offline-demo-stream"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--swanlab", choices=("off", "offline", "cloud", "required"),
                        default="offline")
    parser.add_argument("--require-swanlab", action="store_true",
                        help="fail instead of continuing if SwanLab cannot be initialised")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = replace(config, logging=replace(config.logging, swanlab=args.swanlab))
    workdir = unique_path(Path(args.workdir).resolve())
    stream = unique_path(Path(args.stream).resolve())
    workdir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(workdir / "effective-config.json", config_to_dict(config))

    environment = collect_environment(baseline_root=args.baseline_root, workdir=workdir,
                                      config=config)
    if environment["errors"]:
        print("environment errors: " + "; ".join(environment["errors"]), file=sys.stderr)
        return 1
    if args.require_swanlab and args.swanlab != "off" and not environment["swanlab"]["installed"]:
        print("SwanLab is required but not installed; install "
              "harness/requirements-swanlab.txt", file=sys.stderr)
        return 1

    try:
        subprocess.run([
            sys.executable, str(REPO_ROOT / "tools" / "make_demo_stream.py"),
            "--baseline-root", str(Path(args.baseline_root).resolve()),
            "--output", str(stream), "--batches", str(args.batches),
        ], check=True, cwd=str(REPO_ROOT))
    except Exception as exc:  # noqa: BLE001 - report the failure
        print(f"demo stream generation failed: {exc}", file=sys.stderr)
        return 1

    engine = None
    decisions: list[dict] = []
    error = None
    started = time.monotonic()
    try:
        engine = HarnessEngine(workdir=workdir, config=config,
                               baseline_root=args.baseline_root, verbose=True)
        if args.require_swanlab and not engine.swanlab.enabled:
            raise RuntimeError("SwanLab is not operational: "
                               + str(engine.swanlab.disabled_reason))
        swanlab_before = engine.swanlab.describe()
        batches = discover_batches(stream)
        decisions = engine.run(batches[:args.batches])
        status = engine.status()
        swanlab_after = engine.swanlab.describe()
    except Exception as exc:  # noqa: BLE001 - always leave an evidence trail
        error = f"{type(exc).__name__}: {exc}"
        status = engine.status() if engine is not None else {
            "counters": {}, "champion": {}, "batches": []}
        swanlab_before = engine.swanlab.describe() if engine is not None else {}
        swanlab_after = swanlab_before
    wall_seconds = time.monotonic() - started

    report_path = workdir / "offline-report.md"
    _report(workdir=workdir, stream=stream, report_path=report_path,
            environment=environment, decisions=decisions, status=status,
            swanlab_before=swanlab_before, swanlab_after=swanlab_after,
            wall_seconds=wall_seconds, error=error)
    print(f"offline report: {report_path}")
    print(f"workdir: {workdir}")
    print(f"decisions: {len(decisions)}")
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
