# -*- coding: utf-8 -*-
"""Stage 1 of the loop: drift monitoring.

Two independent monitors, because industrial drift is only visible when you
watch both the inputs and the outcomes:

* ``input``  - per-feature two-sample Kolmogorov-Smirnov statistic against the
  training reference, backed by the population-stability index for the report.
  KS is bounded in [0, 1] and does not explode on the heavy-tailed spectral
  features the way PSI does.
* ``output`` - rolling error of the deployed model when labels have arrived, and
  the prediction-distribution mean when they have not.

The output is one ``DriftSignal`` with level ``ok`` / ``warn`` / ``alarm``.
Only ``alarm`` may spend budget.
"""
from dataclasses import dataclass, field

import numpy as np

BINS = 10


def _psi(reference, current, bins=BINS):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_p = np.maximum(ref_hist / max(ref_hist.sum(), 1), 1e-3)
    cur_p = np.maximum(cur_hist / max(cur_hist.sum(), 1), 1e-3)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _ks(reference, current):
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    cur = np.sort(np.asarray(current, dtype=np.float64))
    grid = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, grid, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, grid, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


@dataclass
class DriftSignal:
    level: str = "ok"
    reason: str = ""
    score: float = 0.0
    top_features: list = field(default_factory=list)
    share_shifted: float = 0.0
    performance: dict = field(default_factory=dict)

    def to_dict(self):
        return {"level": self.level, "reason": self.reason, "score": round(self.score, 4),
                "top_features": self.top_features, "share_shifted": round(self.share_shifted, 4),
                "performance": self.performance}


class DriftMonitor:
    def __init__(self, reference_x, paths, settings):
        self.reference = np.asarray(reference_x, dtype=np.float64)
        self.paths = list(paths)
        self.cfg = settings
        self.history = []
        self._short_names = [self._name(p) for p in self.paths]

    @staticmethod
    def _name(path_string):
        import json
        try:
            keys = [value for kind, value in json.loads(path_string) if kind == "key"]
        except Exception:  # noqa: BLE001
            keys = []
        return ".".join(keys[-2:]) if keys else path_string[:24]

    def detect(self, batch_x, expected=None, predicted=None):
        batch_x = np.asarray(batch_x, dtype=np.float64)
        ks = np.array([_ks(self.reference[:, j], batch_x[:, j]) for j in range(batch_x.shape[1])])
        psi = np.array([_psi(self.reference[:, j], batch_x[:, j]) for j in range(batch_x.shape[1])])
        threshold = float(self.cfg["ks_alarm"])
        share = float(np.mean(ks > threshold))
        top = sorted(range(len(ks)), key=lambda i: -ks[i])[:5]
        top_features = [{"feature": self._short_names[i], "ks": round(float(ks[i]), 3),
                         "psi": round(float(psi[i]), 2)} for i in top]

        signal = DriftSignal(score=float(ks.max()), top_features=top_features, share_shifted=share)
        performance = {}
        if expected is not None and predicted is not None and len(expected):
            error = np.asarray(predicted, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
            performance = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
                           "mean_pred": float(np.mean(predicted)), "mean_target": float(np.mean(expected))}
        elif predicted is not None and len(predicted):
            performance = {"mean_pred": float(np.mean(predicted))}
        signal.performance = performance

        if share >= float(self.cfg["share_alarm"]) or ks.max() >= float(self.cfg["ks_alarm"]):
            signal.level = "alarm"
            signal.reason = (f"{int(round(share * 100))}% of features exceed KS {threshold} "
                             f"(max {ks.max():.2f})")
        elif share >= float(self.cfg["share_warn"]) or ks.max() >= float(self.cfg["ks_warn"]):
            signal.level = "warn"
            signal.reason = f"moderate input shift (max KS {ks.max():.2f})"
        else:
            signal.level = "ok"
            signal.reason = "inputs stay inside the training envelope"
        self.history.append(signal.to_dict())
        return signal
