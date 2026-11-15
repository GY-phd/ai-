"""Persistent harness state (champion lineage, counters, batch ledger)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .config import config_to_dict
from .util import load_json, now_utc, write_json_atomic

STATE_FORMAT = "harness.state.v1"
STATE_FILE = "state.json"


class StateError(ValueError):
    pass


def new_state(
    *,
    baseline_root: str,
    profile: str,
    baseline_checkpoint_sha256: str,
    champion_checkpoint: str,
    champion_sha256: str,
    champion_meta: dict[str, Any],
    public_validation_metrics: dict[str, Any],
    guard_metrics: dict[str, Any],
    guard_info: dict[str, Any],
    replay_info: dict[str, Any] | None,
    config,
) -> dict[str, Any]:
    timestamp = now_utc()
    return {
        "format": STATE_FORMAT,
        "harness_version": __version__,
        "created_at": timestamp,
        "updated_at": timestamp,
        "baseline": {
            "root": baseline_root,
            "profile": profile,
            "official_checkpoint_sha256": baseline_checkpoint_sha256,
        },
        "config": config_to_dict(config),
        "data": {"guard": guard_info, "replay": replay_info},
        "champion": {
            "stage_id": "official-baseline",
            "checkpoint": champion_checkpoint,
            "sha256": champion_sha256,
            "generation": 0,
            "is_static": True,
            "promoted_at_batch": None,
            "metrics": {"batch": public_validation_metrics, "guard": guard_metrics},
            "reference": {
                "batch": public_validation_metrics,
                "guard": guard_metrics,
                "source": "public_validation",
            },
            "last_metrics": None,
            "model": champion_meta,
        },
        "previous_champion": None,
        "counters": {
            "attempts": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "promotions": 0,
            "rollbacks": 0,
            "holds": 0,
            "errors": 0,
        },
        "batches": [],
        "events": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    state = load_json(path)
    if not isinstance(state, dict) or state.get("format") != STATE_FORMAT:
        raise StateError("unsupported or missing harness state format")
    for key in ("baseline", "config", "data", "champion", "counters", "batches", "events"):
        if key not in state:
            raise StateError(f"harness state is missing {key!r}")
    champion = state["champion"]
    if not isinstance(champion, dict) or not isinstance(champion.get("checkpoint"), str):
        raise StateError("harness state champion is invalid")
    if not isinstance(champion.get("generation"), int) or champion["generation"] < 0:
        raise StateError("harness state champion generation is invalid")
    if not isinstance(state["batches"], list) or not isinstance(state["events"], list):
        raise StateError("harness state ledgers are invalid")
    counters = state["counters"]
    if not isinstance(counters, dict) or any(not isinstance(counters.get(name), int) or counters[name] < 0
                                            for name in ("attempts", "successful_experiments",
                                                         "failed_experiments", "promotions",
                                                         "rollbacks", "holds", "errors")):
        raise StateError("harness state counters are invalid")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_utc()
    write_json_atomic(path, state)


def processed_batches(state: dict[str, Any]) -> set[str]:
    return {str(record.get("batch_id")) for record in state["batches"] if isinstance(record, dict)}


def record_batch(state: dict[str, Any], record: dict[str, Any], *, replace: bool = False) -> None:
    batch_id = str(record.get("batch_id"))
    if replace:
        state["batches"] = [item for item in state["batches"] if item.get("batch_id") != batch_id]
    state["batches"].append(record)


def add_event(state: dict[str, Any], event: str, **details: Any) -> None:
    state["events"].append({"ts": now_utc(), "event": event, **details})


def champion_summary(champion: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": champion.get("stage_id"),
        "generation": champion.get("generation"),
        "is_static": bool(champion.get("is_static")),
        "checkpoint": champion.get("checkpoint"),
        "sha256": champion.get("sha256"),
        "metrics": champion.get("metrics"),
        "reference": champion.get("reference"),
    }
