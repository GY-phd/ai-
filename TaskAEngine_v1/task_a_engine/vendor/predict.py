"""Portable Task A prediction entry point, used inside an exported model bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .baselines.model import Model
from .baselines.data import json_value
from .baselines.predict import predict as format_prediction
import numpy as np

from . import feature_ops as features

MODES = ("stable", "fusion", "witness_veto", "raw_support", "projected_support")


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def design(values, center, scale):
    z = np.clip((values - center) / scale, -6, 6)
    return np.column_stack((np.ones(len(z)), z))


class Predictor:
    def __init__(self, manifest_path, variant=None):
        manifest_path = Path(manifest_path).resolve()
        self.folder = manifest_path.parent
        self.manifest = json_value(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["format"] != "a.portable-bundle.v1":
            raise ValueError("unsupported bundle format")
        for relative, expected in self.manifest["files"].items():
            path = (self.folder / relative).resolve()
            if not path.is_relative_to(self.folder) or not path.is_file() or digest(path) != expected:
                raise ValueError(f"bundle file failed integrity check: {relative}")
        self.variant = variant or self.manifest["active_variant"]
        if self.variant not in MODES:
            raise ValueError("unknown model variant")
        self.parent = Model.load(self.folder / self.manifest["incumbent"])
        self.meta = self.parent.meta
        if self.meta["classes"] != self.manifest["classes"] or len(self.meta["paths"]) != 128:
            raise ValueError("bundle class order or feature schema mismatch")
        self.focus = self.meta["classes"].index("Acetaldehyde")
        self.fusion = self.state = None
        if self.variant != "stable":
            with np.load(self.folder / self.manifest["fusion"], allow_pickle=False) as data:
                self.fusion = {name: data[name] for name in data.files}
            f = self.fusion
            if (str(f["parent_digest"]) != digest(self.folder / self.manifest["incumbent"])
                    or str(f["view"]) != "kinetics" or str(f["method"]) != "linear" or float(f["weight"]) != .5):
                raise ValueError("fusion parent or architecture mismatch")
        if self.variant in MODES[2:]:
            with np.load(self.folder / self.manifest["routing_state"], allow_pickle=False) as data:
                self.state = {name: data[name] for name in data.files}

    def predict(self, x):
        base = self.parent.predict(x)
        if self.variant == "stable":
            return base
        f = self.fusion
        kinetics = features.transform(x, "kinetics")
        auxiliary = features.softmax(6 * (design(kinetics, f["mean"], f["scale"]) @ f["coef"]))
        fused = (base + auxiliary) / 2
        if self.variant == "fusion":
            return fused
        s = self.state
        rows = np.flatnonzero((fused.argmax(axis=1) == self.focus) & (base.argmax(axis=1) != self.focus))
        if len(rows):
            values = features.transform(x[rows], "rising_over_steady")
            margins = design(values, s["witness_center"], s["witness_scale"]) @ s["witness_coef"]
            rejected = rows[margins <= .5]
            fused[rejected] = base[rejected]
        if self.variant in MODES[3:]:
            prefix = self.variant + "_"
            values = features.project(kinetics, True) if self.variant == "projected_support" else kinetics
            z = (values - s[prefix + "center"]) / s[prefix + "scale"]
            basis = s[prefix + "basis"]
            residual = np.mean((z - (z @ basis) @ basis.T) ** 2, axis=1)
            fallback = residual > s[prefix + "cutoff"]
            fused[fallback] = base[fallback]
        return fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--variant", choices=MODES)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    model = Predictor(args.manifest, args.variant)
    if args.verify:
        print(json.dumps({"status": "verified", "variant": model.variant, "classes": model.meta["classes"]}))
        return
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json_value(line)
        if (not isinstance(request, dict) or set(request) - {"task", "inputs"} or "inputs" not in request
                or request.get("task", "classification") != "classification"):
            raise ValueError("prediction requires inputs and accepts optional classification task")
        print(json.dumps(format_prediction(model, request["inputs"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
