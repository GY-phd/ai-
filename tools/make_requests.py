# -*- coding: utf-8 -*-
"""Turn a labelled JSONL file into platform-shaped prediction requests (drop ``answer``)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = 0
    with Path(args.data).open(encoding="utf-8") as src, Path(args.output).open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps({"task": "regression", "inputs": row["inputs"]}, ensure_ascii=False) + "\n")
            rows += 1
    print(f"wrote {rows} requests to {args.output}")


if __name__ == "__main__":
    main()
