#!/usr/bin/env python3
"""Run the offline drift-stream benchmark and compare with the static baseline.

Every batch is engine-disjoint: batch training uses public train engines and
batch validation uses public validation engines; units never repeat within a
split.  The report measures per-batch and cumulative RMSE gain/regret of the
selected Harness model against the official static checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.benchmark import batch_result, summarize  # noqa: E402
from harness.config import config_to_dict, load_config  # noqa: E402
from harness.doctor import collect_environment  # noqa: E402
from harness.engine import HarnessEngine, discover_batches  # noqa: E402
from harness.util import now_utc, unique_path, write_json_atomic  # noqa: E402

DEFAULT_BASELINE = REPO_ROOT / "cpu-c-guide-20261110-v1"
DEFAULT_CONFIG = REPO_ROOT / "harness" / "configs" / "default.json"


def _fmt(value, digits=3, suffix=""):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _report(path: Path, *, environment: dict, manifest: dict, config: dict,
            rows: list[dict], summary: dict, status: dict, swanlab: dict,
            wall_seconds: float, error: str | None) -> None:
    lines = [
        "# Offline Drift-stream Benchmark",
        "",
        f"- generated_at: {now_utc()}",
        f"- stream: `{manifest.get('source', {}).get('train')}` / "
        f"`{manifest.get('source', {}).get('validation')}`",
        f"- batches: {summary.get('batches', 0)}",
        f"- wall_seconds: {wall_seconds:.2f}",
        f"- config: profile={config.get('profile')} metric={config.get('metric')}",
        "",
        "## Environment",
        "",
        f"- Python {environment['python']}, NumPy {environment['numpy']}, "
        f"{environment['cpu_count']} CPU cores / {environment['cpu_threads_requested']} requested",
        f"- GPU required: {environment['gpu_required']}; network required: {environment['network_required']}",
        f"- baseline checkpoint: `{environment['baseline']['checkpoint_sha256']}`",
        f"- SwanLab: installed={environment['swanlab']['installed']} "
        f"mode={swanlab.get('mode')} enabled={swanlab.get('enabled')} "
        f"run_dir=`{swanlab.get('run_dir')}`",
        "",
        "## Per-batch results",
        "",
        "| batch | subset | drift | severity | decision | static RMSE | selected RMSE | gain | "
        "static guard | selected guard | exp |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    metadata = {item["batch_id"]: item for item in manifest.get("batches", [])}
    for row in rows:
        meta = metadata.get(row["batch_id"], {})
        lines.append(
            f"| {row['batch_id']} | {meta.get('source_subset')} | {meta.get('drift_type')} | "
            f"{row['drift_severity']} | {row['decision']} | {_fmt(row['static_rmse'])} | "
            f"{_fmt(row['selected_rmse'])} | {_fmt((row['gain_rel'] or 0) * 100, 2, '%')} | "
            f"{_fmt(row['static_guard_rmse'])} | {_fmt(row['selected_guard_rmse'])} | "
            f"{row['experiments']} |")
    lines += [
        "",
        "## Aggregate",
        "",
        f"- mean RMSE: static {_fmt(summary['mean_static_rmse'])}, "
        f"harness {_fmt(summary['mean_selected_rmse'])}",
        f"- mean relative gain: {_fmt(summary['mean_gain_rel'] * 100, 2, '%')}; "
        f"median {_fmt(summary['median_gain_rel'] * 100, 2, '%')}",
        f"- net RMSE gain: {_fmt(summary['net_rmse_gain'])}; "
        f"cumulative gain {_fmt(summary['cumulative_gain'])}, "
        f"cumulative regret {_fmt(summary['cumulative_regret'])}",
        f"- improved / equal / worse batches: {summary['improved_batches']} / "
        f"{summary['equal_batches']} / {summary['worse_batches']}",
        f"- static-floor violations: {summary['static_floor_violations']}",
        f"- decisions: " + ", ".join(f"{key}={value}" for key, value in summary["decisions"].items()),
        f"- experiments/attempts: {summary['attempts']} over {summary['batches']} batches "
        f"({summary['attempts_per_batch']:.2f}/batch); gate passes: {summary['gate_passes']}",
        f"- gain per attempt: {_fmt(summary['gain_per_attempt'])} RMSE",
        f"- training seconds: {_fmt(summary['training_seconds'], 2)}; "
        f"wall seconds: {_fmt(summary['wall_seconds'], 2)}",
        "- counters: " + ", ".join(f"{key}={value}" for key, value in (status.get("counters") or {}).items()),
        "",
        "## Engine-disjoint evidence",
        "",
        "Units are allocated without reuse per split; train rows come from public train "
        "engines and validation rows from public validation engines.",
        "",
    ]
    processed = {row["batch_id"] for row in rows}
    for item in manifest.get("batches", []):
        if item["batch_id"] not in processed:
            continue
        lines.append(f"- {item['batch_id']}: train units {item['train_units']}, "
                     f"validation units {item['validation_units']}, "
                     f"rows {item['train_rows']}/{item['validation_rows']}")
    lines += [
        "",
        "## Notes",
        "",
        "- `domain-shift` / `domain-return` use real FD001–FD004 distribution changes.",
        "- `sensor-calibration` and `concept-shift` are synthetic stress cases and are marked "
        "in `batch.json`/`manifest.json`.",
        "- The static checkpoint is evaluated on the same batch rows as the Harness-selected "
        "model; positive gain means lower RMSE than the static floor.",
        "",
    ]
    if error:
        lines += ["## Error", "", f"```\n{error}\n```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE))
    parser.add_argument("--workdir", default=str(REPO_ROOT / "runs" / "drift-benchmark"))
    parser.add_argument("--stream", default=str(REPO_ROOT / "runs" / "drift-stream"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--plan", help="optional drift plan JSON for the stream generator")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-units", type=int, default=6)
    parser.add_argument("--validation-units", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=2000)
    parser.add_argument("--max-validation-rows", type=int, default=500)
    parser.add_argument("--swanlab", choices=("off", "offline", "cloud", "required"),
                        default="off")
    parser.add_argument("--require-swanlab", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")

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
        print("SwanLab is required but not installed", file=sys.stderr)
        return 1

    generator = [
        sys.executable, str(REPO_ROOT / "tools" / "make_drift_stream.py"),
        "--baseline-root", str(Path(args.baseline_root).resolve()),
        "--output", str(stream), "--seed", str(args.seed),
        "--train-units", str(args.train_units),
        "--validation-units", str(args.validation_units),
        "--max-train-rows", str(args.max_train_rows),
        "--max-validation-rows", str(args.max_validation_rows),
    ]
    if args.plan:
        generator += ["--plan", str(Path(args.plan).resolve())]
    try:
        subprocess.run(generator, check=True, cwd=str(REPO_ROOT))
    except Exception as exc:  # noqa: BLE001 - report the failure
        print(f"drift stream generation failed: {exc}", file=sys.stderr)
        return 1

    manifest = json.loads((stream / "manifest.json").read_text(encoding="utf-8"))
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
        swanlab = engine.swanlab.describe()
        batches = discover_batches(stream)
        decisions = engine.run(batches, max_batches=args.max_batches)
        status = engine.status()
    except Exception as exc:  # noqa: BLE001 - always leave evidence
        error = f"{type(exc).__name__}: {exc}"
        status = engine.status() if engine is not None else {
            "counters": {}, "champion": {}, "batches": []}
        swanlab = engine.swanlab.describe() if engine is not None else {}
    wall_seconds = time.monotonic() - started

    rows = [batch_result(record) for record in decisions]
    summary = None
    summary_error = None
    try:
        summary = summarize(rows, wall_seconds=wall_seconds)
    except Exception as exc:  # noqa: BLE001 - partial runs are still reported
        summary_error = f"{type(exc).__name__}: {exc}"
        summary = {
            "batches": len(rows), "comparable_batches": 0, "attempts": 0,
            "gate_passes": 0, "decisions": {}, "drift_severities": {},
            "improved_batches": 0, "equal_batches": 0, "worse_batches": 0,
            "static_floor_violations": 0, "mean_static_rmse": None,
            "median_static_rmse": None, "mean_selected_rmse": None,
            "median_selected_rmse": None, "mean_gain_rel": 0.0,
            "median_gain_rel": 0.0, "best_gain_rel": 0.0, "worst_gain_rel": 0.0,
            "net_rmse_gain": 0.0, "cumulative_gain": 0.0, "cumulative_regret": 0.0,
            "gain_per_attempt": 0.0, "training_seconds": 0.0,
            "wall_seconds": wall_seconds, "attempts_per_batch": 0.0,
        }
    payload = {
        "format": "harness.drift-benchmark.v1",
        "generated_at": now_utc(),
        "workdir": str(workdir),
        "stream": str(stream),
        "config": config_to_dict(config),
        "environment": environment,
        "manifest": manifest,
        "rows": rows,
        "summary": summary,
        "status": status,
        "swanlab": swanlab,
        "wall_seconds": wall_seconds,
        "error": error,
        "summary_error": summary_error,
    }
    write_json_atomic(workdir / "benchmark.json", payload)
    report_path = workdir / "benchmark-report.md"
    _report(report_path, environment=environment, manifest=manifest,
            config=config_to_dict(config), rows=rows, summary=summary,
            status=status, swanlab=swanlab, wall_seconds=wall_seconds,
            error=error or summary_error)
    print(f"benchmark report: {report_path}")
    print(f"workdir: {workdir}")
    print(f"batches: {summary['batches']}  attempts: {summary['attempts']}  "
          f"mean gain: {summary['mean_gain_rel'] * 100:.2f}%  "
          f"net RMSE gain: {summary['net_rmse_gain']:.3f}")
    if error or summary_error:
        print(f"error: {error or summary_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
