"""Run simulated drift streams through the harness, optionally with a planner model.

This is the data-collection front half of the pipeline. Without ``--llm`` it
simply produces more rule-planner decision records for ``build_dataset``. With
``--llm`` it points the harness at PyTRIO's OpenAI-compatible endpoint, so the
base model proposes trials, the harness executes them, and the decision records
become outcome-labelled distillation data.

Example:
    export PYTRIO_API_KEY=...
    python -m train.run_streams --out work/sft/streams \
        --seeds 7,11,23,31,47,61 --scenarios sensor-aging,regime --workers 4

    python -m train.run_streams --out work/sft/teacher \
        --seeds 7,23,47 --scenarios sensor-aging,regime --llm \
        --model Qwen/Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], log: Path, env: dict) -> tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return process.returncode, str(log)


def one_job(job: tuple[str, int], args, env: dict) -> dict:
    scenario, seed = job
    root = Path(args.out) / f"{scenario}-seed{seed}"
    stream_dir = root / "stream"
    run_dir = root / "run"
    if args.skip_existing and (run_dir / "state.json").is_file():
        return {"job": f"{scenario}-{seed}", "status": "skipped"}

    python = args.python
    if not (stream_dir / "stream").is_dir():
        code, log = _run(
            [python, "-m", "harness", "simulate", "--public", args.public,
             "--out", str(stream_dir), "--scenario", scenario, "--seed", str(seed)],
            root / "simulate.log", env,
        )
        if code:
            return {"job": f"{scenario}-{seed}", "status": "simulate_failed", "log": log}

    cmd = [python, "-m", "harness", "run", "--stream", str(stream_dir / "stream"),
           "--workdir", str(run_dir), "--swanlab-mode", "disabled"]
    if args.llm:
        cmd.append("--llm")
        if args.extra_body:
            config = root / "llm-config.json"
            config.write_text(json.dumps({"llm": {"extra_body": json.loads(args.extra_body)}}), encoding="utf-8")
            cmd += ["--config", str(config)]
    code, log = _run(cmd, root / "run.log", env)
    if code:
        return {"job": f"{scenario}-{seed}", "status": "run_failed", "log": log}
    return {"job": f"{scenario}-{seed}", "status": "ok", "run": str(run_dir), "log": log}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--public", default="cpu-c-guide-20261110-v1/public")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="7,11,23,31,47,61")
    parser.add_argument("--scenarios", default="sensor-aging,regime")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--llm", action="store_true", help="enable the OpenAI-compatible planner")
    parser.add_argument("--model", default=os.environ.get("PYTRIO_MODEL", "Qwen/Qwen3.8-27B"))
    parser.add_argument("--base-url", default=os.environ.get("PYTRIO_BASE_URL", "https://pytrio.com/api/openai/v1"))
    parser.add_argument("--extra-body", default="", help="JSON object merged into the chat-completions body")
    args = parser.parse_args()

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.llm:
        if not env.get("PYTRIO_API_KEY") and not Path.home().joinpath(".pytrio").exists():
            raise SystemExit("--llm needs PYTRIO_API_KEY or a saved `trio login` profile")
        env["PYTRIO_BASE_URL"] = args.base_url
        env["PYTRIO_MODEL"] = args.model

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    scenarios = [token.strip() for token in args.scenarios.split(",") if token.strip()]
    jobs = [(scenario, seed) for scenario in scenarios for seed in seeds]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(one_job, job, args, env): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"jobs": len(jobs), "ok": sum(r["status"] == "ok" for r in results),
                      "skipped": sum(r["status"] == "skipped" for r in results),
                      "failed": sum(r["status"].endswith("failed") for r in results),
                      "out": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
