"""Shared schema for experiment-planner SFT data.

The runtime prompt lives in ``harness/llm.py``. Training must reproduce it
exactly, otherwise the fine-tuned model sees a different distribution than it
will at inference time. Keep every prompt/validation rule here so the dataset
builder, the teacher runner, and the evaluator agree.
"""
from __future__ import annotations

import json

from harness.config import ExperimentConfig
from harness.experiment import sanitize
from harness.llm import SYSTEM, extract_json

SYSTEM_PROMPT = SYSTEM
FIELDS = ("data_scope", "replay_ratio", "learning_rate", "batch_size", "epochs")


def user_prompt(context: dict, count: int) -> str:
    """Mirror ``harness.llm.propose`` byte for byte."""
    return f"请提出至多 {count} 个试验。上下文：\n" + json.dumps(context, ensure_ascii=False)


def action_space_from_config(cfg: ExperimentConfig | None = None) -> dict:
    cfg = cfg or ExperimentConfig()
    return {
        "data_scope": ["recent", "replay", "cumulative"],
        "replay_ratio": [0.1, 2.0],
        "learning_rate": [cfg.lr_min, cfg.lr_max],
        "batch_size": list(cfg.batch_sizes),
        "epochs": [1, cfg.max_epochs],
    }


def _within(value, low, high) -> bool:
    try:
        return float(low) <= float(value) <= float(high)
    except (TypeError, ValueError):
        return False


def validate_response(
    text: str, context: dict, count: int, cfg: ExperimentConfig | None = None
) -> tuple[dict | None, list[str]]:
    """Parse and constrain a planner reply.

    Returns ``(normalized_reply, errors)``. A non-empty error list means the
    reply should not be used as an SFT label. This intentionally rejects
    out-of-space values instead of clamping them: the runtime sanitizer clamps,
    but a label that needed clamping teaches the model values it should not
    have emitted.
    """
    cfg = cfg or ExperimentConfig()
    space = context.get("action_space") or action_space_from_config(cfg)
    errors: list[str] = []

    try:
        parsed = extract_json(text)
    except Exception as error:  # ValueError from harness.llm
        return None, [f"json: {type(error).__name__}: {error}"]
    if not isinstance(parsed, dict):
        return None, ["json: top level is not an object"]

    analysis = parsed.get("analysis")
    if not isinstance(analysis, str):
        analysis = ""

    raw_trials = parsed.get("trials")
    if not isinstance(raw_trials, list):
        return None, ["trials: missing list"]

    base = sanitize(context.get("base_trial") or {}, cfg)
    if base is None:
        return None, ["context.base_trial: not sanitizable"]

    kept: list[dict] = []
    seen: set[tuple] = set()
    for index, raw in enumerate(raw_trials):
        if not isinstance(raw, dict):
            errors.append(f"trials[{index}]: not an object")
            continue
        scope = raw.get("data_scope")
        if scope not in space["data_scope"]:
            errors.append(f"trials[{index}].data_scope: {scope!r} outside {space['data_scope']}")
            continue
        checks = (
            ("replay_ratio", space["replay_ratio"]),
            ("learning_rate", space["learning_rate"]),
            ("epochs", space["epochs"]),
        )
        bad = False
        for field, (low, high) in checks:
            if field in raw and not _within(raw[field], low, high):
                errors.append(f"trials[{index}].{field}: {raw[field]!r} outside [{low}, {high}]")
                bad = True
        batch = raw.get("batch_size", 256)
        try:
            batch = int(batch)
        except (TypeError, ValueError):
            errors.append(f"trials[{index}].batch_size: {batch!r} is not an integer")
            bad = True
        else:
            if batch not in space["batch_size"]:
                errors.append(f"trials[{index}].batch_size: {batch} outside {space['batch_size']}")
                bad = True
        if bad:
            continue

        config = sanitize(raw, cfg)
        if config is None:
            errors.append(f"trials[{index}]: sanitize returned None")
            continue
        key = tuple(config[field] for field in FIELDS)
        if key in seen:
            errors.append(f"trials[{index}]: duplicate of an earlier trial")
            continue
        if all(config[field] == base[field] for field in FIELDS):
            errors.append(f"trials[{index}]: identical to base_trial")
            continue
        seen.add(key)
        hypothesis = raw.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            hypothesis = "对照试验"
        kept.append({"hypothesis": hypothesis.strip()[:400], **config})

    if not kept:
        return None, errors or ["trials: no valid trial"]
    if len(kept) > count:
        kept = kept[:count]
    return {"analysis": analysis.strip()[:500], "trials": kept}, errors


def load_jsonl(path) -> list[dict]:
    import pathlib

    rows = []
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
