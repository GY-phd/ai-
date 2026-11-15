"""One JSON request and response per line for a local Task A Engine."""

import argparse
import json
import sqlite3
import sys

from . import Engine
from .vendor.baselines.data import json_value

METHODS = {"predict", "diagnose", "propose", "train", "evaluate", "activate", "rollback", "status"}


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    args = parser.parse_args()
    engine = Engine(args.store, mode=args.mode)
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json_value(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            action = request.get("action")
            if action not in METHODS or set(request) - {"id", "action", "args"}:
                raise ValueError("unknown action or envelope field")
            result = getattr(engine, action)(**request.get("args", {}))
            response = {"id": request_id, "ok": True, "result": result}
        except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as error:
            response = {"id": request_id, "ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
