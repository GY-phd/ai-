# -*- coding: utf-8 -*-
"""Stage 2: turn a drift signal into a diagnosis card the planner can act on.

The distinction that matters operationally:

* ``covariate`` - inputs moved, but the input->target relation looks intact.
  Cheap fixes (input calibration, head refit) may be enough; large annotation
  spend is *not* justified.
* ``concept``   - inputs moved and the error grew more than the shift predicts.
  The relation itself changed; only labelled refresh can help.
* ``prior``     - the model's output level is systematically off while the
  within-batch ordering still tracks the target.  This is the failure mode of
  this dataset: the fleet lifetime prior does not match the incoming bearings.
* ``data``      - degenerate batch (constant channels, spikes, missing bands).
  Correct action is to flag and not retrain.
* ``none``      - nothing worth spending on.
"""
from dataclasses import dataclass, field


@dataclass
class DiagnosisCard:
    drift_type: str
    confidence: float
    top_features: list = field(default_factory=list)
    needs_labels: bool = False
    evidence: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self):
        return {"drift_type": self.drift_type, "confidence": round(self.confidence, 2),
                "top_features": self.top_features, "needs_labels": self.needs_labels,
                "evidence": self.evidence, "summary": self.summary}


def _is_degenerate(batch_x):
    std = batch_x.std(axis=0)
    return float((std < 1e-9).mean()) > 0.5


def diagnose(signal, batch_x, reference_perf=None, cold_start=False):
    psi_max = signal.score
    share = signal.share_shifted
    perf = signal.performance or {}
    has_labels = "rmse" in perf

    if cold_start:
        return DiagnosisCard("cold_start", 0.95, signal.top_features, False,
                             {"psi_max": psi_max, "share_shifted": share,
                              "deployed": "official_static_baseline"},
                             "the deployed model has never been adapted; measure a trained candidate before judging the stream")

    if _is_degenerate(batch_x):
        return DiagnosisCard("data", 0.9, signal.top_features, False,
                             {"degenerate_features": True},
                             "batch looks like a sensor/data fault, not a distribution shift")

    if signal.level == "ok" and not has_labels:
        return DiagnosisCard("none", 0.9, signal.top_features, False, {"psi_max": psi_max},
                             "no actionable drift")

    if not has_labels:
        # Label-free: we can only claim an input-side story.
        if share >= 0.5:
            return DiagnosisCard("covariate", 0.5, signal.top_features, False,
                                 {"psi_max": psi_max, "share_shifted": share},
                                 "broad input shift without delayed labels; input calibration is the only safe move")
        return DiagnosisCard("covariate", 0.4, signal.top_features, False,
                             {"psi_max": psi_max, "share_shifted": share},
                             "localised input shift; monitor before spending")

    rmse, mae = perf["rmse"], perf["mae"]
    bias = perf["mean_pred"] - perf["mean_target"]
    ratio = rmse / max(reference_perf["rmse"], 1e-9) if reference_perf else 1.0
    bias_share = abs(bias) / max(rmse, 1e-9)

    if ratio >= 1.15 and psi_max >= 0.25:
        return DiagnosisCard("concept", min(0.5 + ratio / 4, 0.9), signal.top_features, True,
                             {"rmse_ratio": round(ratio, 3), "bias": round(bias, 1), "psi_max": round(psi_max, 3)},
                             "error grew faster than the input shift, the input->target relation moved")
    if bias_share >= 0.6 and ratio >= 1.05:
        return DiagnosisCard("prior", min(0.55 + bias_share / 2, 0.95), signal.top_features, False,
                             {"rmse_ratio": round(ratio, 3), "bias": round(bias, 1), "bias_share": round(bias_share, 3)},
                             "model output level is off while the ordering still tracks the target (fleet prior mismatch)")
    if ratio >= 1.05:
        return DiagnosisCard("covariate", 0.6, signal.top_features, False,
                             {"rmse_ratio": round(ratio, 3), "psi_max": round(psi_max, 3)},
                             "error tracks a measurable input shift")
    return DiagnosisCard("none", 0.7, signal.top_features, False,
                         {"rmse_ratio": round(ratio, 3)},
                         "performance is inside tolerance despite the alarm")
