"""The Harness entry point.  The platform appends describe, train or predict.

The SDK's recommended fixed entry:

    python3 /app/harness.py describe
    python3 /app/harness.py train
    python3 /app/harness.py predict

This file is the SDK's own ``template/harness.py``.  It is copied verbatim
so that the image's entry point is exactly what the SDK documents, and so
that upgrading the SDK does not silently change the entry point.
"""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
