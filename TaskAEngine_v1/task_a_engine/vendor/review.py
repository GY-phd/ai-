"""Evaluate one frozen model variant on paired data and write a model-update decision."""

import argparse
import json
from pathlib import Path

from .baselines.data import read_dataset
from .predict import Predictor, digest
from .decision import assess
from . import feature_ops
import numpy as np

PUBLIC_BATCH_2 = "sha256:6be1370388bf67d017f7426a51e149a65baf9bc1186f5f41f6731ce5006ffbee"


def metrics(y, probabilities):
    result = feature_ops.score(y, probabilities)
    observed = [c["f1"] for c in result["per_class"].values() if c["support"] or c["predicted"]]
    result["macro_f1"] = float(np.mean(observed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--variant", default="stable", choices=("stable", "fusion", "witness_veto", "raw_support", "projected_support"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--evidence", choices=("public_development", "independent_released"), default="public_development")
    parser.add_argument("--release-id")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = Predictor(args.manifest, "stable")
    policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else None
    assessed = None
    if args.variant == "stable" or args.evaluation is None:
        result = assess(None, {} if args.variant == "stable" else {args.variant: None}, evidence="unavailable", thresholds=policy)
    else:
        if args.train is None or digest(args.train) != base.manifest["training_receipts"]["train_sha256"]:
            raise ValueError("supply the original Batch 1 training file for overlap verification")
        if args.evidence == "independent_released" and not (args.release_id or "").strip():
            raise ValueError("independent evaluation requires a release identifier")
        train = read_dataset(args.train, "classification", base.meta["classes"], base.meta["paths"])
        data = read_dataset(args.evaluation, "classification", base.meta["classes"], base.meta["paths"])
        if set(train["ids"]) & set(data["ids"]) or set(map(tuple, train["x"])) & set(map(tuple, data["x"])):
            raise ValueError("training and evaluation rows overlap")
        if args.evidence == "independent_released" and (data["sha256"] == PUBLIC_BATCH_2 or
                any(identity.startswith("gas_b02_") for identity in data["ids"])):
            raise ValueError("public Batch 2 is development evidence")
        challenger = Predictor(args.manifest, args.variant)
        old, new = metrics(data["y"], base.predict(data["x"])), metrics(data["y"], challenger.predict(data["x"]))
        result = assess(old, {args.variant: new}, evidence=args.evidence, thresholds=policy)
        result["per_class_and_confusion"] = {"stable": old, args.variant: new}
        assessed = {"sha256": data["sha256"], "rows": len(data["y"]), "release_id": args.release_id}
    result["evaluation"] = assessed
    result["manifest_sha256"] = digest(args.manifest)
    result["recommended_variant"] = result["selected_candidate"] or "stable"
    result["active_variant"] = base.manifest["active_variant"]
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result["agent_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
