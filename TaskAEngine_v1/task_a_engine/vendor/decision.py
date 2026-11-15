"""Compare Task A scores and apply a separate local model-update policy."""

from numbers import Integral, Real

if __package__:
    from . import rollout
else:
    import rollout


ETHANOL = "Ethanol"
DEFAULT_THRESHOLDS = {
    "min_macro_f1_gain": rollout.MIN_MACRO_GAIN,
    "min_acetaldehyde_precision": rollout.MIN_ACETALDEHYDE_PRECISION,
    "max_other_recall_drop": rollout.MAX_OTHER_RECALL_DROP,
}
EVIDENCE_TYPES = {"unavailable", "source_holdout", "public_development", "independent_released"}


def rate(value):
    if isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1:
        return float(value)
    return None


def competition_scores(metrics):
    return {key: rate((metrics or {}).get(key)) for key in ("macro_f1", "accuracy")}


def guardrails(incumbent, candidate, thresholds):
    if incumbent is None or candidate is None:
        return {"passed": False, "delta_macro_f1": None,
                "violations": [{"criterion": "missing_paired_evaluation"}]}

    violations = []
    old_f1 = rate(incumbent.get("macro_f1"))
    new_f1 = rate(candidate.get("macro_f1"))
    delta = None if old_f1 is None or new_f1 is None else new_f1 - old_f1
    if delta is None:
        violations.append({"criterion": "missing_macro_f1"})
    elif delta + 1e-12 < thresholds["min_macro_f1_gain"]:
        violations.append({"criterion": "min_macro_f1_gain", "observed": delta})

    old_classes = incumbent.get("per_class") or {}
    new_classes = candidate.get("per_class") or {}
    for label in rollout.CLASSES:
        old = old_classes.get(label) or {}
        new = new_classes.get(label) or {}
        support = old.get("support")
        other_support = new.get("support")
        if (not all(isinstance(n, Integral) and not isinstance(n, bool) for n in (support, other_support))
                or support < 0 or support != other_support):
            violations.append({"criterion": "missing_or_mismatched_support", "class": label})
            continue
        if support == 0:
            if label in (ETHANOL, rollout.ACETALDEHYDE):
                violations.append({"criterion": "missing_required_class", "class": label})
            continue
        if label == rollout.ACETALDEHYDE:
            old_f1, new_f1, precision = (rate(old.get("f1")), rate(new.get("f1")), rate(new.get("precision")))
            if None in (old_f1, new_f1, precision):
                violations.append({"criterion": "missing_acetaldehyde_metrics", "class": label})
                continue
            if new_f1 <= old_f1:
                violations.append({"criterion": "acetaldehyde_f1_must_improve", "class": label})
            if precision + 1e-12 < thresholds["min_acetaldehyde_precision"]:
                violations.append({"criterion": "min_acetaldehyde_precision", "class": label, "observed": precision})
        else:
            before, after = rate(old.get("recall")), rate(new.get("recall"))
            if before is None or after is None:
                violations.append({"criterion": "missing_recall", "class": label})
                continue
            drop = before - after
            if drop > thresholds["max_other_recall_drop"] + 1e-12:
                violations.append({"criterion": "max_other_recall_drop", "class": label, "observed": drop})

    return {"passed": not violations, "delta_macro_f1": delta, "violations": violations}


def assess(incumbent, candidates, *, evidence="source_holdout", thresholds=None):
    """Review paired metrics. Independent evidence attests paired, legally released held-out labels."""
    if evidence not in EVIDENCE_TYPES:
        raise ValueError("unknown evaluation evidence type")
    policy = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if set(policy) != set(DEFAULT_THRESHOLDS) or any(rate(value) is None for value in policy.values()):
        raise ValueError("unknown policy threshold or threshold outside [0, 1]")
    policy = {name: float(value) for name, value in policy.items()}
    candidates = candidates or {}
    scores = {name: competition_scores(metrics) for name, metrics in candidates.items()}
    reviews = {name: guardrails(incumbent, metrics, policy) for name, metrics in candidates.items()}
    ranked = sorted((name for name, metrics in scores.items() if metrics["macro_f1"] is not None),
                    key=lambda name: (-scores[name]["macro_f1"], name))
    best_scored = next(iter(ranked), None)
    best_safe = next((name for name in ranked if reviews[name]["passed"]), None)
    if not candidates:
        reason = "no_candidate"
    elif evidence == "unavailable" or incumbent is None or best_scored is None:
        reason = "missing_paired_evaluation"
    elif best_safe is None:
        reason = "guardrails_failed"
    elif evidence != "independent_released":
        reason = "independent_evaluation_required"
    else:
        reason = "passed_independent_evaluation"
    selected = best_safe if reason == "passed_independent_evaluation" else None
    decision = "promote" if selected is not None else "keep"
    reviewed = best_safe if best_safe is not None else best_scored
    violations = reviews[reviewed]["violations"] if reviewed is not None else []
    summary = {"decision": decision, "reason": reason, "evidence": evidence, "candidate": reviewed,
               "delta_macro_f1": reviews[reviewed]["delta_macro_f1"] if reviewed is not None else None,
               "failed_checks": [{key: v[key] for key in ("criterion", "class") if key in v} for v in violations]}
    return {"decision": decision, "reason": reason, "evidence": evidence,
            "selected_candidate": selected, "best_scored_candidate": best_scored, "best_safe_candidate": best_safe,
            "competition": {"main_metric": "macro_f1", "secondary_metric": "accuracy",
                            "incumbent": competition_scores(incumbent), "candidates": scores},
            "safety": {"thresholds": policy, "candidates": reviews}, "agent_summary": summary}
