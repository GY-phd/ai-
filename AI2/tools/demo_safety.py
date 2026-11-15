# -*- coding: utf-8 -*-
"""Safety demo: prove that a bad candidate is rejected and that rollback is byte-exact.

Run after (or instead of) a harness run:

    python tools/demo_safety.py --workspace runs/harness-b

It takes the current champion, builds a deliberately damaged candidate
(output layer scaled by 3x), pushes it through the same gate the loop uses, and
then exercises the rollback action.  Exit code is 0 only when both behave.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import load as load_config  # noqa: E402
from harness.experiments import ExperimentRunner  # noqa: E402
from harness.gate import evaluate as gate_evaluate  # noqa: E402
from harness.ledger import Ledger  # noqa: E402
from harness.util import sha256_file  # noqa: E402
from rulkit.data import read_dataset  # noqa: E402
from rulkit.model import Model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="runs/harness-b")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    workspace = Path(args.workspace)
    ledger = Ledger(workspace)
    champion = ledger.champion()
    if champion is None:
        raise SystemExit("no champion in this workspace; run the harness first")

    validation = read_dataset(config["data"]["validation"])
    runner = ExperimentRunner(config, read_dataset(config["data"]["train"]), validation, workspace / "safety")

    # 1) a damaged candidate: same weights, output layer scaled by 3
    damaged = Model.load(champion["checkpoint"])
    damaged.params["w2"] = damaged.params["w2"] * 3.0
    damaged.params["b2"] = damaged.params["b2"] * 3.0
    damaged.validate()
    damaged_path = workspace / "safety" / "damaged_candidate.npz"
    damaged_path.parent.mkdir(parents=True, exist_ok=True)
    if damaged_path.exists():
        damaged_path.unlink()
    damaged.save(damaged_path)
    damaged_metrics = runner._metrics(damaged_path)  # noqa: SLF001 - diagnostic reuse

    gate = gate_evaluate(champion["metrics"], damaged_metrics, config["gate"], {"train_seconds_left": 900})
    print(json.dumps({"champion_rmse": round(champion["metrics"]["rmse"], 2),
                      "damaged_rmse": round(damaged_metrics["rmse"], 2),
                      "gate": gate.to_dict()}, ensure_ascii=False, indent=2))

    # 2) rollback restores the champion byte-for-byte
    result = runner.run("rollback", {}, champion, previous_champion=champion)
    same = sha256_file(result["checkpoint"]) == champion["checkpoint_sha256"]
    print(json.dumps({"rollback_checkpoint": result["checkpoint"],
                      "byte_identical": same,
                      "rmse_after_rollback": round(result["metrics"]["rmse"], 2)}, ensure_ascii=False, indent=2))

    ok = gate.decision == "reject" and same
    print("SAFETY OK" if ok else "SAFETY CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
