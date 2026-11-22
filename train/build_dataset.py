"""Build SFT examples for the harness experiment planner from harness runs.

Each training row is exactly what the runtime planner sees:

    system  = harness.llm.SYSTEM
    user    = "请提出至多 N 个试验。上下文：\\n" + <the same JSON context>
    assistant = {"analysis": ..., "trials": [...]}

The context is reconstructed from a decision record plus the trial digest that
existed before that decision. Labels come from trials the harness actually ran
(``source="rule"``, ``source="llm"``, or both). By default the best non-base
trial(s) of the round are used; refuted trials are only used when nothing else
exists.

Usage:
    python -m train.build_dataset \
        --runs runs work/sft/run-* \
        --out train/data/planner-train.jsonl \
        --val-out train/data/planner-val.jsonl \
        --val-runs sensor-aging-61,regime-61
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from harness.config import load_config
from harness.experiment import sanitize
from train.schema import (
    FIELDS,
    SYSTEM_PROMPT,
    action_space_from_config,
    user_prompt,
)


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_config(run_dir: Path):
    path = run_dir / "config.json"
    if path.is_file():
        try:
            return load_config(str(path)).experiment
        except Exception:
            pass
    return load_config(None).experiment


def _decisions(run_dir: Path) -> list[dict]:
    rows = []
    for path in (run_dir / "decisions").glob("*.json"):
        rows.append(_load(path))
    rows.sort(key=lambda row: int(row.get("batch_index", 0)))
    return rows


def _digest_entry(decision: dict) -> dict | None:
    trials = (decision.get("experiments") or {}).get("trials") or []
    if not trials:
        return None
    diagnosis = decision.get("diagnosis") or {}
    return {
        "batch": decision.get("batch"),
        "drift": diagnosis.get("kind"),
        "severity": diagnosis.get("severity"),
        "action": decision.get("action"),
        "trials": [
            {
                key: trial.get(key)
                for key in (
                    "arm",
                    "data_scope",
                    "learning_rate",
                    "batch_size",
                    "epochs",
                    "select_score",
                    "rel_gain_vs_control",
                    "verdict",
                )
            }
            for trial in trials
        ],
    }


def _incumbent(decision: dict) -> str:
    rollback = decision.get("rollback") or {}
    if rollback.get("triggered") and rollback.get("to"):
        return str(rollback["to"])
    return str(decision.get("previous_version") or "v0")


def _budget(decision: dict, cfg) -> int:
    diagnosis = decision.get("diagnosis") or {}
    rollback = decision.get("rollback") or {}
    routine = not (diagnosis.get("flag") or rollback.get("triggered"))
    if routine:
        return 2
    return 4 if diagnosis.get("severity") == "severe" else 3


def _context(decision: dict, history: list[dict], cfg) -> dict:
    diagnosis = decision.get("diagnosis") or {}
    performance = (decision.get("drift") or {}).get("performance") or {}
    policy_before = (decision.get("experiments") or {}).get("policy_before") or {}
    metrics = ((decision.get("after") or {}).get("metrics") or {})
    kind = "regression" if "rmse" in metrics else "classification"
    primary = "rmse，越低越好" if kind == "regression" else "error_rate，越低越好"
    return {
        "task": kind,
        "primary_metric": primary,
        "diagnosis": {
            key: diagnosis.get(key)
            for key in ("kind", "severity", "summary", "top_features", "error_profile")
        },
        "online_model": {
            "version": _incumbent(decision),
            "score_on_new_batch": performance.get("score"),
            "score_when_promoted": performance.get("anchor"),
        },
        "base_trial": sanitize(policy_before, cfg),
        "has_history": int(decision.get("batch_index", 0)) > 0,
        "history": history[-4:],
        "action_space": action_space_from_config(cfg),
    }


def _trial_config(trial: dict, cfg) -> dict | None:
    return sanitize(trial, cfg)


def _rank(trial: dict) -> tuple:
    gain = trial.get("rel_gain_vs_control")
    try:
        gain = float(gain)
    except (TypeError, ValueError):
        gain = -math.inf
    score = trial.get("select_score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = math.inf
    return (trial.get("verdict") == "refuted", gain <= 0, score)


def _hypothesis(trial: dict) -> str:
    text = trial.get("hypothesis")
    if isinstance(text, str) and text.strip():
        return text.strip()[:400]
    changed = []
    for field in FIELDS:
        value = trial.get(field)
        if value is not None and field in ("data_scope", "learning_rate", "batch_size", "epochs"):
            changed.append(f"{field}={value}")
    return "对照试验：" + ("，".join(changed) if changed else "改变训练配置")


def _analysis(decision: dict, kept: list[dict]) -> str:
    diagnosis = decision.get("diagnosis") or {}
    head = diagnosis.get("meaning") or diagnosis.get("summary") or diagnosis.get("kind") or "数据状态未知"
    head = str(head).strip().rstrip("，。； ")
    if len(head) > 160:
        head = head[:160].rstrip("，。； ")
    reasons = "；".join(item["hypothesis"][:80].rstrip("。；， ") for item in kept[:2])
    return f"{head}。拟做 {len(kept)} 个对照试验：{reasons}。"


def _select_trials(decision: dict, cfg, source: str, count: int) -> list[dict]:
    trials = (decision.get("experiments") or {}).get("trials") or []
    candidates = [trial for trial in trials if trial.get("arm") != "base"]
    if source == "llm":
        pool = [trial for trial in candidates if trial.get("source") == "llm"]
    elif source == "rule":
        pool = [trial for trial in candidates if trial.get("source") != "llm"]
    else:  # auto: prefer model proposals when the run used a model
        llm = [trial for trial in candidates if trial.get("source") == "llm"]
        pool = llm or candidates

    # A refuted trial is a control the harness explicitly rejected. Only fall
    # back to it when the round produced nothing else. The same holds for
    # non-positive gains when at least one trial improved on the control.
    usable = [trial for trial in pool if trial.get("verdict") != "refuted"]
    pool = usable or pool
    positive = [trial for trial in pool if float(trial.get("rel_gain_vs_control") or 0) > 0]
    pool = positive or pool
    pool = sorted(pool, key=_rank)

    kept: list[dict] = []
    seen: set[tuple] = set()
    for trial in pool:
        config = _trial_config(trial, cfg)
        if config is None:
            continue
        key = tuple(config[field] for field in FIELDS)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"hypothesis": _hypothesis(trial), **config,
                     "_arm": trial.get("arm"), "_source": trial.get("source") or "rule",
                     "_verdict": trial.get("verdict"),
                     "_rel_gain": trial.get("rel_gain_vs_control"),
                     "_select_score": trial.get("select_score")})
        if len(kept) >= count:
            break
    return kept


def build_run(run_dir: Path, source: str) -> tuple[list[dict], dict]:
    cfg = _run_config(run_dir)
    decisions = _decisions(run_dir)
    run_name = run_dir.parent.name if run_dir.name == "run" else run_dir.name
    examples: list[dict] = []
    digest: list[dict] = []
    skipped = 0

    for decision in decisions:
        history = digest[-4:]
        trials = (decision.get("experiments") or {}).get("trials") or []
        if trials:
            count = max(1, _budget(decision, cfg) - 1)
            kept = _select_trials(decision, cfg, source, count)
            if kept:
                context = _context(decision, history, cfg)
                target = {
                    "analysis": _analysis(decision, kept),
                    "trials": [{key: item[key] for key in ("hypothesis", *FIELDS)} for item in kept],
                }
                kept_sources = {item["_source"] for item in kept}
                example_source = "llm" if kept_sources == {"llm"} else ("rule" if kept_sources == {"rule"} else "mixed")
                examples.append(
                    {
                        "id": f"{run_name}-{decision.get('batch')}",
                        "run": run_name,
                        "batch": decision.get("batch"),
                        "batch_index": int(decision.get("batch_index", 0)),
                        "source": example_source,
                        "count": count,
                        "system": SYSTEM_PROMPT,
                        "user": user_prompt(context, count),
                        "assistant": json.dumps(target, ensure_ascii=False),
                        "meta": {
                            "action": decision.get("action"),
                            "diagnosis": (decision.get("diagnosis") or {}).get("kind"),
                            "severity": (decision.get("diagnosis") or {}).get("severity"),
                            "arms": [item["_arm"] for item in kept],
                            "verdicts": [item["_verdict"] for item in kept],
                            "rel_gains": [item["_rel_gain"] for item in kept],
                            "select_scores": [item["_select_score"] for item in kept],
                        },
                    }
                )
            else:
                skipped += 1
            entry = _digest_entry(decision)
            if entry is not None:
                digest.append(entry)
    return examples, {"run": run_name, "decisions": len(decisions), "examples": len(examples), "skipped": skipped}


def _split(examples: list[dict], val_runs: list[str], val_fraction: float):
    if val_runs:
        train, val = [], []
        for row in examples:
            (val if any(token in row["run"] for token in val_runs) else train).append(row)
        return train, val
    runs = sorted({row["run"] for row in examples})
    if not runs or val_fraction <= 0:
        return examples, []
    count = max(1, int(round(len(runs) * val_fraction))) if len(runs) > 1 else 0
    chosen = set(runs[:: max(1, len(runs) // count)][:count]) if count else set()
    train = [row for row in examples if row["run"] not in chosen]
    val = [row for row in examples if row["run"] in chosen]
    return train, val


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="harness run directories (contain decisions/ and config.json)")
    parser.add_argument("--out", required=True, help="training JSONL")
    parser.add_argument("--val-out", help="validation JSONL; omitted with --no-split")
    parser.add_argument("--label-source", choices=("rule", "llm", "auto"), default="auto")
    parser.add_argument("--val-runs", default="", help="comma-separated run-name substrings for validation")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="run-level validation fraction when --val-runs is empty")
    parser.add_argument("--no-split", action="store_true", help="write every example to --out")
    args = parser.parse_args()

    all_rows: list[dict] = []
    stats = []
    for raw in args.runs:
        run_dir = Path(raw)
        if not (run_dir / "decisions").is_dir():
            print(f"skip {run_dir}: no decisions/")
            continue
        rows, info = build_run(run_dir, args.label_source)
        all_rows.extend(rows)
        stats.append(info)

    if args.no_split or not args.val_out:
        train, val = all_rows, []
    else:
        names = [token.strip() for token in args.val_runs.split(",") if token.strip()]
        train, val = _split(all_rows, names, args.val_fraction)

    _write(Path(args.out), train)
    if val:
        _write(Path(args.val_out), val)

    print(json.dumps(
        {
            "runs": stats,
            "examples": len(all_rows),
            "train": len(train),
            "val": len(val),
            "sources": {name: sum(row["source"] == name for row in all_rows) for name in ("rule", "llm", "mixed")},
            "out": str(args.out),
            "val_out": str(args.val_out) if val else None,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
