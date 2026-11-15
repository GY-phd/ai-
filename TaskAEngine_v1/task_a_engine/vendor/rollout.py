"""Review a candidate on paired six-class gas recognition metrics."""

from numbers import Integral, Real


MIN_MACRO_GAIN = 0.005
MIN_ACETALDEHYDE_PRECISION = 0.65
MAX_OTHER_RECALL_DROP = 0.03
ACETALDEHYDE = "Acetaldehyde"
CLASSES = ("Ethanol", "Ethylene", "Ammonia", ACETALDEHYDE, "Acetone", "Toluene")


def _is_rate(value):
    return isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1


def review(incumbent, candidate, *, deployment_evidence=False):
    """Promote only on qualifying metrics from independent, released evaluation.

    The caller must supply paired metrics for the same evaluation rows and can
    attest deployment_evidence only when the evaluation was independently
    released and held out of development and candidate selection.
    """
    incumbent = incumbent or {}
    candidate = candidate or {}
    violations = []
    old_macro = incumbent.get("macro_f1")
    new_macro = candidate.get("macro_f1")
    delta = None
    if _is_rate(old_macro) and _is_rate(new_macro):
        delta = float(new_macro - old_macro)
        if delta + 1e-12 < MIN_MACRO_GAIN:
            violations.append({"criterion": "min_macro_f1_gain", "observed": delta})
    else:
        violations.append({"criterion": "missing_macro_f1"})

    old_classes = incumbent.get("per_class") or {}
    new_classes = candidate.get("per_class") or {}
    for label in CLASSES:
        old = old_classes.get(label) or {}
        new = new_classes.get(label) or {}
        support = old.get("support")
        new_support = new.get("support")
        if (not isinstance(support, Integral) or isinstance(support, bool)
                or not isinstance(new_support, Integral) or isinstance(new_support, bool)
                or support < 0 or support != new_support):
            violations.append({"criterion": "missing_or_mismatched_support", "class": label})
            continue
        if label == ACETALDEHYDE:
            old_f1, new_f1 = old.get("f1"), new.get("f1")
            precision = new.get("precision")
            if not support or not all(_is_rate(rate) for rate in (old_f1, new_f1, precision)):
                violations.append({"criterion": "missing_acetaldehyde_evidence", "class": label})
                continue
            if new_f1 <= old_f1:
                violations.append({"criterion": "acetaldehyde_f1_must_improve", "class": label})
            if precision < MIN_ACETALDEHYDE_PRECISION:
                violations.append({"criterion": "min_acetaldehyde_precision", "class": label,
                                   "observed": float(precision)})
        elif support:
            old_recall, new_recall = old.get("recall"), new.get("recall")
            if not _is_rate(old_recall) or not _is_rate(new_recall):
                violations.append({"criterion": "missing_recall", "class": label})
                continue
            drop = float(old_recall - new_recall)
            if drop > MAX_OTHER_RECALL_DROP + 1e-12:
                violations.append({"criterion": "max_other_recall_drop", "class": label,
                                   "observed": drop})

    criteria_passed = not violations
    independently_evaluated = deployment_evidence is True
    if not criteria_passed:
        decision, reason = "keep", "guardrails_failed"
    elif not independently_evaluated:
        decision, reason = "keep", "independent_evaluation_required"
    else:
        decision, reason = "promote", "passed_independent_evaluation"
    return {
        "decision": decision,
        "reason": reason,
        "criteria_passed": criteria_passed,
        "deployment_evidence": independently_evaluated,
        "delta_macro_f1": delta,
        "thresholds": {
            "min_macro_f1_gain": MIN_MACRO_GAIN,
            "min_acetaldehyde_precision": MIN_ACETALDEHYDE_PRECISION,
            "max_other_recall_drop": MAX_OTHER_RECALL_DROP,
        },
        "violations": violations,
    }
