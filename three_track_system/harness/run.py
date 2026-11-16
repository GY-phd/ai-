# -*- coding: utf-8 -*-
"""CLI entry point: ``python -m harness.run --config harness/config/harness.json``."""
import sys

from .orchestrator import main

if __name__ == "__main__":
    sys.exit(main())
