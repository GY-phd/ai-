# -*- coding: utf-8 -*-
"""Measure the **deployed** model's floor, the way the platform scores it.

Why this module exists
-----------------------
The platform's objective score has a ``worst_stage`` line (5 points,
"漂移流中表现最差阶段的底线能力"), and it read **0.182 / 5** on the
2026-11-25 run.  The score is a property of the **deployed** model at
each stage of the drift stream, and the deployed model only changes when
a candidate passes the gate - so it cannot be measured on a single
candidate.  It has to be measured on the **trajectory** the rehearsal
actually produced.

This module takes an offline rehearsal's ``run_report.json``, replays the
adoption decisions to reconstruct the deployed model at every round, and
reports:

* the deployed metric at every round (the trajectory),
* the **worst** value on that trajectory (the floor),
* the floor read on the two stage groupings of profile C - the five
  **lifecycle stages** and the four **operating conditions**,
* the same readings for the **static baseline**, so the floor can be
  compared against it.

The comparison that decides the line
------------------------------------
The platform scores the floor of the drift stream.  A candidate only
wins that line if its floor is **at least as good as** the static
baseline's floor - no matter how good the headline number is.  So the
report ends with that comparison, per grouping, and with the deployed
trajectory's own worst round.

Run::

    python -m research.tracks.c_deployed --run runs/harness-offline
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines.model import Model  # noqa: E402
from research.tracks.c_protocol import load  # noqa: E402
from research.tracks.c_stage import STAGES, stage_masks  # noqa: E402

#: The static baseline's public validation score, per direction.
BASELINE = {"A": 0.7240, "B": 681.160, "C": 52.8686}


def group_readings(y, pred, lifecycle, conditions, stage_names, condition_names):
    """RMSE per group, plus the worst group and the full split."""
    out = {"full": float(np.sqrt(np.mean((pred - y) ** 2)))}
    for name in stage_names:
        mask = lifecycle == name
        out[name] = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))) if mask.any() else float("nan")
    for name in condition_names:
        mask = conditions == name
        out[name] = float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))) if mask.any() else float("nan")
    life = [out[n] for n in stage_names if out[n] == out[n]]
    cond = [out[n] for n in condition_names if out[n] == out[n]]
    out["worst_lifecycle"] = float(max(life)) if life else float("nan")
    out["worst_condition"] = float(max(cond)) if cond else float("nan")
    out["worst"] = float(max(out["worst_lifecycle"], out["worst_condition"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=str(ROOT / "runs" / "harness-offline"))
    ap.add_argument("--package", default="")
    ap.add_argument("--out",
                    default=str(ROOT / "artifacts" / "tracks" / "c_deployed.json"))
    args = ap.parse_args(argv)

    run = pathlib.Path(args.run)
    report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
    ctx = load(args.package or None)
    masks = stage_masks(ctx)
    y = np.asarray(ctx["valid"]["y"], dtype=np.float64)
    lifecycle = np.zeros(len(ctx["vunit"]), dtype=object)
    for index, mask in enumerate(masks):
        lifecycle[mask] = "stage%d" % (index + 1)
    stage_names = ["stage%d" % (i + 1) for i in range(STAGES)]
    conditions = np.asarray(ctx["vsub"], dtype=object)
    condition_names = sorted(set(ctx["vsub"].tolist()))

    base_model = Model.load(ctx["pkg"] / "checkpoint.npz")
    base = group_readings(y, np.asarray(base_model.predict(ctx["valid"]["x"])),
                          lifecycle, conditions, stage_names, condition_names)
    print("static baseline")
    print("  %s" % " ".join("%s=%.4f" % (k, v) for k, v in base.items()))
    print("  worst lifecycle %.4f   worst condition %.4f   full %.4f"
          % (base["worst_lifecycle"], base["worst_condition"], base["full"]))

    rows = []
    print("\n=== track C: the deployed trajectory ===")
    for record in report["datasets"]["C"]["rounds"]:
        checkpoint = run / "checkpoints" / ("C-round-%d.npz" % record["round"])
        deployed = record["adopted"]
        if not deployed or not checkpoint.is_file():
            # The parent is kept, so the deployed model is the last
            # adopted candidate - which the previous rows already carry.
            readings = rows[-1]["readings"] if rows else base
            print("r%-3d kept the parent   full %8.4f  worst life %8.4f  worst cond %8.4f"
                  % (record["round"], readings["full"],
                     readings["worst_lifecycle"], readings["worst_condition"]))
            rows.append({"round": record["round"], "adopted": False,
                         "readings": readings})
            continue
        model = Model.load(checkpoint)
        pred = np.asarray(model.predict(ctx["valid"]["x"]), dtype=np.float64)
        readings = group_readings(y, pred, lifecycle, conditions,
                                  stage_names, condition_names)
        print("r%-3d adopted  full %8.4f  worst life %8.4f  worst cond %8.4f  %s"
              % (record["round"], readings["full"], readings["worst_lifecycle"],
                 readings["worst_condition"],
                 " ".join("%s=%.2f" % (n, readings[n]) for n in stage_names)))
        rows.append({"round": record["round"], "adopted": True,
                     "readings": readings})

    # ---- the floor of the deployed trajectory -------------------------
    print("\n=== the floor of the deployed trajectory ===")
    deployed_rows = [r for r in rows if r["adopted"]]
    for key, label in (("full", "full split"),
                       ("worst_lifecycle", "worst lifecycle stage"),
                       ("worst_condition", "worst operating condition")):
        floor = min(r["readings"][key] for r in deployed_rows)
        print("  %-26s floor %8.4f   baseline %8.4f   delta %+8.4f"
              % (label, floor, base[key], floor - base[key]))
    final = deployed_rows[-1]["readings"]
    print("  %-26s final %8.4f   baseline %8.4f   delta %+8.4f"
          % ("final deployed", final["full"], base["full"], final["full"] - base["full"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "format": "contest.research.c-deployed.v1",
        "run": str(run),
        "baseline": base,
        "rounds": rows,
        "floors": {
            key: min(r["readings"][key] for r in deployed_rows)
            for key in ("full", "worst_lifecycle", "worst_condition")
        },
        "final": final,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
