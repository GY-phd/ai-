# -*- coding: utf-8 -*-
"""JSONL prediction adapter (platform protocol).

Reads one request per line from stdin, writes one response per line to stdout,
flushing after every row so the platform can score a live stream.

    {"task": "regression", "inputs": {...}}  ->  {"outputs": {"value": 123.45}}

The adapter never reads labels; ``inputs`` is the only field handed to the model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .data import json_value
from .features import flatten
from .model import Model


def predict_one(model, inputs):
    paths, values = flatten(inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    value = model.predict(np.asarray([values], dtype=np.float64))[0]
    return {"outputs": {"value": float(max(0.0, value))}}


def run(checkpoint, source, sink):
    model = Model.load(checkpoint)
    for line in source:
        if not line.strip():
            continue
        request = json_value(line)
        if not isinstance(request, dict) or set(request) - {"task", "inputs"} or "inputs" not in request:
            raise ValueError("prediction request requires only task and inputs")
        if request.get("task", model.meta["task"]) != model.meta["task"]:
            raise ValueError("prediction task mismatch")
        sink.write(json.dumps(predict_one(model, request["inputs"]), allow_nan=False) + "\n")
        sink.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", help="JSONL request file (default: stdin)")
    parser.add_argument("--output", help="JSONL response file (default: stdout)")
    args = parser.parse_args(argv)
    if args.input:
        with Path(args.input).open(encoding="utf-8") as src:
            if args.output:
                with Path(args.output).open("w", encoding="utf-8") as dst:
                    run(args.checkpoint, src, dst)
            else:
                run(args.checkpoint, src, sys.stdout)
    else:
        run(args.checkpoint, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
