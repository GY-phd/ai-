# -*- coding: utf-8 -*-
"""Configuration loading with explicit defaults.

Every path in the config may be absolute or relative to the repository root
(``rul_solution/``).  Nothing here embeds credentials; the language-model
endpoint is always resolved from environment variables so that the package
itself stays free of weights, keys and services.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "run_name": "harness-b",
    "data": {
        "train": "data/public/train.jsonl",
        "validation": "data/public/validation.jsonl",
        "baseline_checkpoint": "data/checkpoint.npz",
    },
    "stream": {
        "path": "data/public/validation.jsonl",
        "batch_size": 200,
        "has_labels": True,
        "label_delay_batches": 0,
    },
    "llm": {
        "model_id": "Qwen3.8-27B",
        "base_url_env": "HARNESS_LLM_BASE_URL",
        "api_key_env": "HARNESS_LLM_API_KEY",
        "timeout_s": 30,
        "max_tokens": 1024,
        "temperature": 0.0,
        "mode": "auto",
    },
    "budget": {
        "max_experiments": 5,
        "max_train_seconds": 900,
        "max_tokens": 20000,
        "max_wall_seconds": 2400,
    },
    "detector": {
        "ks_warn": 0.12,
        "ks_alarm": 0.22,
        "share_warn": 0.15,
        "share_alarm": 0.25,
        "perf_drop_pct": 5.0,
    },
    "bootstrap": {"mode": "baseline_only"},
    "gate": {"min_gain_pct": 1.0, "max_local_regression_pct_of_global": 20.0},
    "train": {
        "hidden": 96,
        "epochs": 400,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "batch_size": 128,
        "feature_transform": "eng34",
        "transfer_epochs": 80,
        "seed": 42,
    },
    "search": {
        "hidden": [96],
        "epochs": [400, 600],
        "weight_decay": [0.01],
        "init": ["reinit", "transfer"],
        "max_trials": 3,
    },
    "workspace": "runs/harness",
}


def _merge(base, override):
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def data_root():
    """Where the official data pack lives; override with RUL_DATA_ROOT."""
    return Path(os.environ.get("RUL_DATA_ROOT", str(ROOT / "data")))


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def resolve_data(path):
    """Resolve a config path, rebasing the ``data/`` prefix onto RUL_DATA_ROOT."""
    text = str(path).replace("\\", "/")
    if text.startswith("data/"):
        return str(data_root() / text[len("data/"):])
    return str(resolve(text))


def load(path=None, overrides=None):
    config = json.loads(json.dumps(DEFAULTS))
    if path:
        config = _merge(config, json.loads(Path(path).read_text(encoding="utf-8")))
    if overrides:
        config = _merge(config, overrides)
    config["workspace"] = str(resolve(config["workspace"]))
    for key in ("train", "validation", "baseline_checkpoint"):
        config["data"][key] = resolve_data(config["data"][key])
    config["stream"]["path"] = resolve_data(config["stream"]["path"])
    config["llm"]["base_url"] = os.environ.get(config["llm"]["base_url_env"], "").strip()
    config["llm"]["api_key"] = os.environ.get(config["llm"]["api_key_env"], "").strip()
    return config
