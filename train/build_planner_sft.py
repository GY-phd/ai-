# -*- coding: utf-8 -*-
"""Build ``state -> action`` SFT data for the 27B harness decision layer.

Data sources, all local to this repository:

* ``evidence/ledger.jsonl`` + ``evidence/hypothesis_chain.jsonl``: the nine
  real decisions the harness made on the B-direction stream.  These are the
  gold examples.
* ``harness/policy.py``: the deterministic rule table.  Counterfactual
  diagnosis cards are sampled over the same field space and labelled by the
  policy, which expands nine decisions into a balanced training set without
  inventing a new oracle.

The rendered ``system``/``user`` strings come from ``harness.llm`` itself
(``SYSTEM_PROMPT``, ``USER_TEMPLATE``, ``compact_history``), and the skills
files are loaded exactly the way ``Harness.__init__`` loads them.  The model
therefore trains on the prompt it will actually see.

Usage:
    python -m train.build_planner_sft --out train/data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from harness.llm import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    compact_history,
    validate_plan,
)
from harness.policy import PolicyPlanner

ROOT = Path(__file__).resolve().parents[1]
DRIFT_TYPES = ("cold_start", "data", "none", "prior", "covariate", "concept")

SUMMARY = {
    "cold_start": "the deployed model has never been adapted; measure a trained candidate before judging the stream",
    "data": "batch looks like a sensor/data fault, not a distribution shift",
    "none": "performance is inside tolerance despite the alarm",
    "prior": "model output level is off while the ordering still tracks the target (fleet prior mismatch)",
    "covariate": "error tracks a measurable input shift",
    "concept": "error grew faster than the input shift, the input->target relation moved",
}

FEATURE_POOL = [
    "horizontal.band1_energy_0_2k", "horizontal.band2_energy_2k_6k",
    "horizontal.band3_energy_6k_12k", "horizontal.crest_factor",
    "horizontal.dominant_freq", "horizontal.impulse_factor", "horizontal.kurtosis",
    "horizontal.margin_factor", "horizontal.mean_abs", "horizontal.peak",
    "horizontal.peak_to_peak", "horizontal.rms", "horizontal.shape_factor",
    "horizontal.skewness", "horizontal.spectral_centroid", "horizontal.spectral_kurtosis",
    "horizontal.spectral_rms", "vertical.band1_energy_0_2k",
    "vertical.band2_energy_2k_6k", "vertical.band3_energy_6k_12k",
    "vertical.crest_factor", "vertical.dominant_freq", "vertical.impulse_factor",
    "vertical.kurtosis", "vertical.margin_factor", "vertical.mean_abs", "vertical.peak",
    "vertical.peak_to_peak", "vertical.rms", "vertical.shape_factor",
    "vertical.skewness", "vertical.spectral_centroid", "vertical.spectral_kurtosis",
    "vertical.spectral_rms",
]


def load_skills(root: Path = ROOT) -> str:
    chunks = []
    for path in sorted((root / "harness" / "skills").glob("*.md")):
        chunks.append(f"# {path.stem}\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(chunks)


def render_prompt(card: dict, champion: dict, budget: dict, history: list[dict], skills: str) -> tuple[str, str]:
    system = SYSTEM_PROMPT + ("\n\n" + skills if skills else "")
    user = USER_TEMPLATE.format(
        card=json.dumps(card, ensure_ascii=False, indent=2),
        champion=json.dumps(champion, ensure_ascii=False, indent=2),
        budget=json.dumps(budget, ensure_ascii=False, indent=2),
        history=json.dumps(compact_history(history), ensure_ascii=False, indent=2),
    )
    return system, user


def target_json(plan: dict) -> str:
    clean = {key: plan.get(key) for key in
             ("action", "params", "hypothesis", "expected_gain_pct", "risk", "reason")}
    clean["params"] = clean.get("params") or {}
    clean["expected_gain_pct"] = float(clean.get("expected_gain_pct") or 0.0)
    return json.dumps(clean, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _budget_state(max_experiments=5, max_train_seconds=900.0, max_tokens=20000, max_wall=2400.0):
    return {
        "experiments_left": max_experiments, "train_seconds_left": max_train_seconds,
        "tokens_left": max_tokens, "wall_seconds_left": max_wall,
        "experiments_used": 0, "train_seconds_used": 0.0, "tokens_used": 0,
        "sweep_exhausted": False,
    }


def real_examples(root: Path = ROOT) -> list[dict]:
    ledger = _read_jsonl(root / "evidence" / "ledger.jsonl")
    chain = _read_jsonl(root / "evidence" / "hypothesis_chain.jsonl")
    report_path = root / "evidence" / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    params_by_cycle = {int(item["iteration"]): (item.get("experiment") or {}).get("params") or {}
                       for item in chain if item.get("iteration") is not None}
    baseline = report.get("baseline") or {"rmse": 681.16, "mae": 563.11}
    champion = {"rmse": float(baseline["rmse"]), "mae": float(baseline.get("mae", baseline["rmse"]))}
    budget = _budget_state()
    history: list[dict] = []
    skills = load_skills(root)
    examples = []
    hpo_seen = 0

    for row in sorted((item for item in ledger if item.get("stage") == "decide"),
                      key=lambda item: int(item.get("cycle", 0))):
        cycle = int(row.get("cycle", 0))
        card = row.get("diagnosis") or {}
        if not card.get("drift_type"):
            continue
        plan = {
            "action": row.get("action"),
            "params": params_by_cycle.get(cycle, {}) or {},
            "hypothesis": row.get("hypothesis") or "",
            "expected_gain_pct": float(row.get("expected_gain_pct") or 0.0),
            "risk": row.get("risk") or "medium",
            "reason": row.get("reason") or "",
        }
        try:
            plan = validate_plan({key: value for key, value in plan.items() if key != "_meta"})
        except ValueError:
            pass
        system, user = render_prompt(card, champion, budget, history, skills)
        examples.append({
            "id": f"real-cycle-{cycle:02d}",
            "split": "train",
            "source": "real",
            "system": system,
            "user": user,
            "assistant": target_json(plan),
            "meta": {"cycle": cycle, "action": plan["action"], "decision": row.get("decision"),
                     "outcome_gain_pct": (row.get("gate") or {}).get("gain_pct")},
        })

        # Advance the simulated budget and champion exactly as the loop does.
        trials = row.get("trials") or []
        budget["experiments_used"] += len(trials) if trials else (1 if plan["action"] != "do_nothing" else 0)
        budget["train_seconds_used"] += float(row.get("cost_seconds") or 0.0)
        budget["experiments_left"] = max(0, 5 - budget["experiments_used"])
        budget["train_seconds_left"] = max(0.0, 900.0 - budget["train_seconds_used"])
        budget["wall_seconds_left"] = max(0.0, 2400.0 - budget["train_seconds_used"])
        if plan["action"] == "hpo":
            hpo_seen += 1
        budget["sweep_exhausted"] = hpo_seen >= 2
        if row.get("decision") == "promote" and row.get("candidate_metrics"):
            candidate = row["candidate_metrics"]
            champion = {"rmse": float(candidate["rmse"]), "mae": float(candidate.get("mae", candidate["rmse"]))}
        history.append(row)
    return examples


def _card(rng: random.Random, drift_type: str) -> dict:
    k = rng.randint(3, 5)
    features = rng.sample(FEATURE_POOL, k)
    top = [{"feature": name, "ks": round(rng.uniform(0.08, 0.95), 3),
            "psi": round(rng.uniform(0.05, 3.0), 3)} for name in features]
    if drift_type == "cold_start":
        confidence, needs_labels = 0.95, False
        evidence = {"psi_max": round(rng.uniform(0.3, 1.0), 3),
                    "share_shifted": round(rng.uniform(0.3, 0.9), 3),
                    "deployed": "official_static_baseline"}
    elif drift_type == "data":
        confidence, needs_labels = 0.9, False
        evidence = {"degenerate_features": True}
    elif drift_type == "none":
        confidence, needs_labels = round(rng.uniform(0.65, 0.95), 2), False
        evidence = {"rmse_ratio": round(rng.uniform(0.85, 1.02), 3)}
    elif drift_type == "covariate":
        confidence, needs_labels = round(rng.uniform(0.4, 0.65), 2), False
        evidence = {"rmse_ratio": round(rng.uniform(1.03, 1.14), 3),
                    "psi_max": round(rng.uniform(0.1, 0.6), 3)}
    elif drift_type == "prior":
        bias_share = rng.uniform(0.6, 0.9)
        confidence, needs_labels = round(min(0.95, 0.55 + bias_share / 2), 2), False
        evidence = {"rmse_ratio": round(rng.uniform(1.05, 1.25), 3),
                    "bias": round(rng.uniform(-400, 400), 1),
                    "bias_share": round(bias_share, 3)}
    else:  # concept
        ratio = rng.uniform(1.15, 1.8)
        confidence, needs_labels = round(min(0.9, 0.5 + ratio / 4), 2), True
        evidence = {"rmse_ratio": round(ratio, 3), "bias": round(rng.uniform(-300, 300), 1),
                    "psi_max": round(rng.uniform(0.25, 1.0), 3)}
    return {"drift_type": drift_type, "confidence": confidence, "top_features": top,
            "needs_labels": needs_labels, "evidence": evidence, "summary": SUMMARY[drift_type]}


def _champion(rng: random.Random) -> dict:
    rmse = rng.uniform(150, 1200)
    return {"rmse": round(rmse, 2), "mae": round(rmse * rng.uniform(0.72, 0.95), 2),
            "per_bearing": {name: {"rmse": round(rmse * rng.uniform(0.85, 1.15), 2),
                                   "rows": rng.randint(30, 1600)}
                            for name in ("xjtu_Bearing1_4", "xjtu_Bearing2_4", "xjtu_Bearing3_4")}}


def _budget(rng: random.Random) -> dict:
    experiments = rng.choice((0, 1, 2, 3, 5))
    train_seconds = rng.choice((0, 25, 60, 200, 600, 900))
    return {
        "experiments_left": experiments,
        "train_seconds_left": train_seconds,
        "tokens_left": rng.choice((500, 2000, 8000, 20000)),
        "wall_seconds_left": rng.choice((60, 300, 1200, 2400)),
        "experiments_used": 5 - experiments,
        "train_seconds_used": 900 - train_seconds,
        "tokens_used": 0,
        "sweep_exhausted": rng.random() < 0.25,
    }


def _history(rng: random.Random, drift_type: str, action: str) -> list[dict]:
    patterns = []
    patterns.append([])
    patterns.append([{"cycle": 1, "action": action, "decision": "reject",
                      "hypothesis": "the previous attempt did not clear the gate",
                      "reason": "gain below threshold", "gate": {"decision": "reject", "gain_pct": -1.2},
                      "cost_seconds": 30.0}])
    patterns.append([{"cycle": 1, "action": "hpo", "decision": "reject",
                      "hypothesis": "a sweep was tried and rejected",
                      "reason": "no bearing regression above the limit", "gate": {"decision": "reject"},
                      "cost_seconds": 52.0}])
    patterns.append([{"cycle": 1, "action": "hpo", "decision": "promote",
                      "hypothesis": "the cold-start sweep improved the champion",
                      "reason": "RMSE improved", "gate": {"decision": "promote", "gain_pct": 22.0},
                      "cost_seconds": 48.0}])
    patterns.append([{"cycle": 1, "action": "do_nothing", "decision": "skip",
                      "hypothesis": "no actionable evidence", "reason": "kept the champion",
                      "gate": {"decision": "skip"}, "cost_seconds": 0.0}])
    pattern = rng.choice(patterns)
    if pattern:
        pattern[0]["cycle"] = rng.randint(1, 6)
        if rng.random() < 0.3:
            pattern.append({**pattern[0], "cycle": pattern[0]["cycle"] + 1,
                            "action": rng.choice(("hpo", "retrain_head", "recalibrate_inputs", "do_nothing")),
                            "decision": rng.choice(("promote", "reject", "skip"))})
    return pattern


def synthetic_examples(count: int, seed: int, skills: str) -> list[dict]:
    rng = random.Random(seed)
    policy = PolicyPlanner()
    examples = []
    for index in range(count):
        drift_type = rng.choice(DRIFT_TYPES)
        card = _card(rng, drift_type)
        champion = _champion(rng)
        budget = _budget(rng)
        action = {"cold_start": "hpo", "data": "do_nothing", "none": "do_nothing",
                  "prior": "retrain_head", "covariate": "recalibrate_inputs",
                  "concept": "hpo"}[drift_type]
        history = _history(rng, drift_type, action)
        plan = policy.plan(card, champion, budget, history)
        plan = {key: value for key, value in plan.items() if key != "_meta"}
        system, user = render_prompt(card, champion, budget, history, skills)
        examples.append({
            "id": f"syn-{seed}-{index:04d}",
            "split": "val" if index % 5 == 0 else "train",
            "source": "policy",
            "system": system,
            "user": user,
            "assistant": target_json(plan),
            "meta": {"drift_type": drift_type, "action": plan["action"],
                     "experiments_left": budget["experiments_left"],
                     "train_seconds_left": budget["train_seconds_left"],
                     "sweep_exhausted": budget["sweep_exhausted"]},
        })
    return examples


BALANCED_ACTIONS = ("do_nothing", "hpo", "retrain_head", "recalibrate_inputs", "adapt", "rollback")
BALANCED_TEXT = {
    "do_nothing": ("none", "no actionable evidence; keeping the champion avoids a needless update"),
    "hpo": ("concept", "the input->target relation moved; a sweep over the public data and the stream so far is the only way to find a configuration that captures it"),
    "retrain_head": ("prior", "output level is biased while ordering is intact; a head refit is the cheapest correction"),
    "recalibrate_inputs": ("covariate", "input statistics moved; align them before spending labels"),
    "adapt": ("concept", "the relation moved but the current configuration is still sound; retrain on initial data plus the stream observed so far"),
    "rollback": ("concept", "the deployed candidate regressed on the new batch; the previous champion was better on the same judge slice"),
}
BALANCED_GAIN = {"do_nothing": 0.0, "hpo": 3.0, "retrain_head": 1.5,
                 "recalibrate_inputs": 1.0, "adapt": 2.5, "rollback": 0.0}
BALANCED_RISK = {"do_nothing": "low", "hpo": "high", "retrain_head": "medium",
                 "recalibrate_inputs": "medium", "adapt": "medium", "rollback": "low"}


def balanced_examples(per_action: int, seed: int, skills: str) -> list[dict]:
    """Counterfactuals that cover the full action space.

    The deterministic policy never emits ``adapt`` or ``rollback``, so a
    policy-only dataset would leave two legal actions untaught.  These rows
    are expert-authored demonstrations for the situations in which those
    actions are the right answer; they are labelled ``source="balanced"`` so
    the training report can separate them from the measured policy traces.
    """
    rng = random.Random(seed + 1)
    examples = []
    for action in BALANCED_ACTIONS:
        drift_type, hypothesis = BALANCED_TEXT[action]
        for index in range(per_action):
            card = _card(rng, drift_type)
            if action == "rollback":
                card["summary"] = hypothesis
                card["evidence"] = {"regression_vs_previous_pct": round(rng.uniform(5, 20), 2),
                                    "rmse_ratio": round(rng.uniform(1.1, 1.6), 3)}
            champion = _champion(rng)
            budget = _budget(rng)
            if action == "hpo":
                # A sweep is the right answer only while the search space has
                # something left to try.
                budget.update({"experiments_left": rng.randint(1, 5),
                               "train_seconds_left": rng.choice((120, 300, 600, 900)),
                               "sweep_exhausted": False})
            elif action == "adapt":
                # Same drift family, different budget state: the configuration
                # is already sound and the search space is exhausted, so the
                # cheap fixed-config retrain is the correct move.
                budget.update({"experiments_left": rng.randint(1, 3),
                               "train_seconds_left": rng.choice((120, 300, 600)),
                               "sweep_exhausted": True})
                card["summary"] = ("the relation moved but the current configuration is still sound; "
                                   "retrain on initial data plus the stream observed so far")
                card["evidence"] = {**card["evidence"], "configuration_stable": True}
            elif action in ("retrain_head", "recalibrate_inputs", "rollback"):
                budget.update({"experiments_left": rng.randint(1, 5),
                               "train_seconds_left": rng.choice((60, 120, 300, 900))})

            # Keep the recent-decision block consistent with the label: a
            # rejection of the same action would make the policy ladder (and
            # any reasonable reader of the card) expect a cheaper fallback.
            history = []
            if action == "retrain_head" and rng.random() < 0.5:
                history = [{"cycle": rng.randint(1, 6), "action": "hpo", "decision": "reject",
                            "hypothesis": "the sweep did not clear the gate",
                            "reason": "gain below threshold", "gate": {"decision": "reject", "gain_pct": -1.0},
                            "cost_seconds": 45.0}]
            elif action == "recalibrate_inputs" and rng.random() < 0.5:
                history = [{"cycle": rng.randint(1, 6), "action": "retrain_head", "decision": "reject",
                            "hypothesis": "the head refit did not correct the bias",
                            "reason": "gain below threshold", "gate": {"decision": "reject", "gain_pct": -0.8},
                            "cost_seconds": 3.0}]
            elif action == "rollback":
                history = [{"cycle": rng.randint(1, 6), "action": "hpo", "decision": "promote",
                            "hypothesis": "a candidate was promoted on the previous batch",
                            "reason": "RMSE improved", "gate": {"decision": "promote", "gain_pct": 8.0},
                            "cost_seconds": 40.0}]
            elif action == "do_nothing":
                history = _history(rng, drift_type, action)
            plan = {
                "action": action,
                "params": {"max_trials": rng.choice((2, 3))} if action == "hpo" else {},
                "hypothesis": hypothesis,
                "expected_gain_pct": BALANCED_GAIN[action],
                "risk": BALANCED_RISK[action],
                "reason": f"diagnosis card: {card['drift_type']} -> {action}",
            }
            plan = {key: value for key, value in validate_plan(plan).items() if key != "_meta"}
            system, user = render_prompt(card, champion, budget, history, skills)
            examples.append({
                "id": f"bal-{action}-{index:03d}",
                "split": "val" if index % 5 == 0 else "train",
                "source": "balanced",
                "system": system,
                "user": user,
                "assistant": target_json(plan),
                "meta": {"drift_type": card["drift_type"], "action": action,
                         "experiments_left": budget["experiments_left"],
                         "train_seconds_left": budget["train_seconds_left"]},
            })
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="train/data")
    parser.add_argument("--synthetic", type=int, default=300)
    parser.add_argument("--balanced-per-action", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    out = Path(args.out)
    skills = load_skills()

    real = real_examples()
    synthetic = synthetic_examples(args.synthetic, args.seed, skills)
    balanced = balanced_examples(args.balanced_per_action, args.seed, skills)
    train = [row for row in real + synthetic + balanced if row["split"] == "train"]
    val = [row for row in synthetic + balanced if row["split"] == "val"]

    write_jsonl(out / "planner-train.jsonl", train)
    write_jsonl(out / "planner-val.jsonl", val)
    write_jsonl(out / "planner-real.jsonl", real)

    actions = {}
    for row in train + val:
        action = json.loads(row["assistant"])["action"]
        actions[action] = actions.get(action, 0) + 1
    manifest = {
        "seed": args.seed, "synthetic_requested": args.synthetic,
        "balanced_per_action": args.balanced_per_action,
        "real": len(real), "train": len(train), "val": len(val),
        "sources": {name: sum(row["source"] == name for row in train + val)
                    for name in ("real", "policy", "balanced")},
        "actions": actions,
        "system_chars": len(real[0]["system"]) if real else len(skills),
        "files": [str(out / name) for name in ("planner-train.jsonl", "planner-val.jsonl", "planner-real.jsonl")],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
