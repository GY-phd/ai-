# -*- coding: utf-8 -*-
"""Verify the harness ledger hash chain.

    python tools/verify_ledger.py --ledger evidence/ledger.jsonl

Every record stores ``prev_sha256`` (the previous record's digest, ``"genesis"``
for the first one) and its own ``record_sha256``.  Re-computing the chain proves
no decision was rewritten after the fact.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def canonical(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_of(record):
    return "sha256:" + hashlib.sha256(canonical(record)).hexdigest()


def verify_records(records):
    """Return (ok, message). Pure function so tests and CI can reuse it."""
    if not records:
        return False, "empty ledger"
    previous = "genesis"
    for index, record in enumerate(records):
        if record.get("prev_sha256") != previous:
            return False, (f"BROKEN LINK at seq={record.get('seq')} (line {index + 1}): "
                           f"prev={record.get('prev_sha256')} expected={previous}")
        digest = digest_of(record)
        if digest != record.get("record_sha256"):
            return False, f"TAMPERED at seq={record.get('seq')} (line {index + 1})"
        previous = digest
    decisions = [r for r in records if r.get("stage") == "decide"]
    promoted = [r for r in decisions if r.get("decision") == "promote"]
    return True, (f"OK  {len(records)} records, {len(decisions)} decisions, "
                  f"{len(promoted)} promotions, chain intact\nhead sha256: {previous}")


def load_ledger(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args(argv)
    ok, message = verify_records(load_ledger(args.ledger))
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
