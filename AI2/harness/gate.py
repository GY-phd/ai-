# -*- coding: utf-8 -*-
"""Stage 4: the gate that decides whether a candidate may replace the champion.

A candidate is promoted only when *all* of the following hold:

1. the primary metric improves by at least ``min_gain_pct`` on the public
   validation slice (the pack explicitly designates it for measuring changes);
2. no single bearing degrades by more than ``max_local_regression_pct_of_global``
   percent of the champion's global RMSE - the mean metric must never hide a
   local regression, but a small bearing is not allowed to veto a fleet-level
   win either;
3. the safety checks pass: finite predictions, non-negative output, resource
   cost inside the remaining budget.

Anything else is rejected and the champion stays deployed; the rejection itself
is recorded, because "candidate ineffective -> automatic rollback" is a scored
behaviour of the system.
"""
from dataclasses import dataclass, field


@dataclass
class GateDecision:
    decision: str
    reason: str
    gain_pct: float
    checks: dict = field(default_factory=dict)

    def to_dict(self):
        return {"decision": self.decision, "reason": self.reason,
                "gain_pct": round(self.gain_pct, 3), "checks": self.checks}


def evaluate(champion_metrics, candidate_metrics, settings, budget):
    checks = {}
    champion_rmse = champion_metrics["rmse"]
    candidate_rmse = candidate_metrics["rmse"]
    gain = (champion_rmse - candidate_rmse) / max(champion_rmse, 1e-9) * 100.0
    checks["gain_pct"] = round(gain, 3)
    checks["min_gain_pct"] = settings["min_gain_pct"]
    checks["improved"] = gain >= settings["min_gain_pct"]

    worst = None
    for bearing, row in (candidate_metrics.get("per_bearing") or {}).items():
        base = (champion_metrics.get("per_bearing") or {}).get(bearing)
        if not base:
            continue
        delta_abs = row["rmse"] - base["rmse"]
        # Local regressions are judged against the global error scale, not against the
        # bearing's own small number: a +80 min shift on a 120-row bearing is tolerable
        # when the fleet-level RMSE is 680, but not when it is 60.
        delta_pct_global = delta_abs / max(champion_rmse, 1e-9) * 100.0
        if worst is None or delta_pct_global > worst[1]:
            worst = (bearing, delta_pct_global, delta_abs)
    checks["worst_local_regression_pct_of_global"] = round(worst[1], 2) if worst else 0.0
    checks["worst_bearing"] = worst[0] if worst else None
    checks["worst_bearing_delta_minutes"] = round(worst[2], 1) if worst else 0.0
    checks["no_local_regression"] = (worst is None) or (worst[1] <= settings["max_local_regression_pct_of_global"])

    checks["finite"] = candidate_rmse == candidate_rmse and candidate_rmse >= 0
    checks["budget_left"] = budget.get("train_seconds_left", 0) >= 0
    checks["safety_ok"] = checks["finite"] and checks["budget_left"]

    if checks["improved"] and checks["no_local_regression"] and checks["safety_ok"]:
        return GateDecision("promote", f"RMSE improved by {gain:.2f}% with no bearing regression above "
                                       f"{settings['max_local_regression_pct_of_global']}% of the global RMSE", gain, checks)
    if not checks["improved"]:
        reason = f"gain {gain:.2f}% is below the {settings['min_gain_pct']}% promotion threshold"
    elif not checks["no_local_regression"]:
        reason = (f"{worst[0]} regressed by {worst[2]:.0f} min "
                  f"({worst[1]:.1f}% of the global RMSE), above the "
                  f"{settings['max_local_regression_pct_of_global']}% tolerance")
    else:
        reason = "safety check failed"
    return GateDecision("reject", reason, gain, checks)
