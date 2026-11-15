# -*- coding: utf-8 -*-
"""Deterministic planning policy.

Same action space and same JSON shape as the Qwen planner, but expressed as an
auditable rule table.  It is used (a) when no model endpoint is reachable - the
submission must still close the loop offline - and (b) as the reference the
model's proposals are compared against during review.
"""
from .llm import ALLOWED_ACTIONS, validate_plan

RULES = (
    ("cold_start", "hpo", "the static baseline was never adapted; train and measure a candidate before judging drift"),
    ("data", "do_nothing", "flag the batch for data quality; re-training on a fault is harmful"),
    ("none", "do_nothing", "no actionable evidence; keeping the champion is the cheapest correct action"),
    ("prior", "retrain_head", "output level is biased with ordering intact; a head refit is cheap"),
    ("covariate", "recalibrate_inputs", "input statistics moved; align them before spending labels"),
    ("concept", "hpo", "relation changed; a fresh candidate trained on public data is the only real fix"),
)


class PolicyPlanner:
    name = "policy"

    def __init__(self, settings=None, skills="", memory=None):
        self.memory = memory or []
        self.tokens_used = 0
        self.calls = 0

    def available(self):
        return True

    def plan(self, card, champion, budget, history):
        drift_type = card.get("drift_type", "none")
        action, reason = "do_nothing", "no rule matched"
        for kind, candidate, why in RULES:
            if kind == drift_type:
                action, reason = candidate, why
                break

        # A rejected candidate already left the champion untouched (that *is* the
        # rollback).  Repeating the same expensive action would burn budget for the
        # same evidence, so climb down the cost ladder instead.
        rejected = [(r.get("action"), r.get("decision")) for r in history[-2:]]
        if any(decision == "reject" and tried == action for tried, decision in rejected):
            ladder = {"hpo": "retrain_head", "retrain_head": "recalibrate_inputs",
                      "recalibrate_inputs": "do_nothing"}
            alternative = ladder.get(action, "do_nothing")
            reason += f" | previous {action} was rejected, falling back to {alternative}"
            action = alternative

        if action == "hpo" and budget.get("sweep_exhausted"):
            action, reason = "retrain_head", "search space exhausted; falling back to the cheap head refit"
        if action in ("retrain_head", "hpo", "recalibrate_inputs") and budget.get("experiments_left", 0) <= 0:
            action, reason = "do_nothing", "experiment budget exhausted"
        if budget.get("train_seconds_left", 0) <= 30 and action in ("retrain_head", "hpo"):
            action, reason = "do_nothing", "training budget exhausted"

        plan = {"action": action, "params": {}, "hypothesis": reason,
                "expected_gain_pct": {"rollback": 0.0, "do_nothing": 0.0}.get(action, 2.0),
                "risk": {"hpo": "high", "retrain_head": "medium", "recalibrate_inputs": "medium"}.get(action, "low"),
                "reason": f"rule table: {drift_type} -> {action}",
                "_meta": {"planner": "policy", "model_id": "deterministic-rules"}}
        assert plan["action"] in ALLOWED_ACTIONS
        return validate_plan({k: v for k, v in plan.items() if not k.startswith("_")}) | {"_meta": plan["_meta"]}
