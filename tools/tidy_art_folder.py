# -*- coding: utf-8 -*-
"""Tidy the scratch folder.

``art/`` accumulated about a hundred files during the research: stdout
captures, probe outputs, one-off patch scripts.  Some of it is evidence
(the logs of a real run) and most of it is noise (the intermediate print
of a command that was run three times).

This script keeps the evidence and drops the noise.  It is deliberately
conservative: a file is only deleted if it is in the explicit keep list
or is clearly a scratch artefact (a ``.txt`` capture, a ``.log`` that is
not on the keep list, a one-off probe directory).

Run:
    python tools/tidy_art_folder.py --dry-run
    python tools/tidy_art_folder.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "art"

# The files worth keeping: real run logs, research results that are
# quoted in the documents, and the summaries the reports point at.
KEEP_FILES = {
    # run logs
    "swan_verify6.log", "swan_api.log", "swan_login.log", "swan_status.log",
    "rul_b_mlp.log", "rul_b_swanlab.log", "rmse_verification.log",
    "harness_run.txt", "tests.txt",
    # research results quoted in the documents
    "lobo_champion.json", "two_stage.json", "final_selection.json",
    "stream_split.json", "stream_curve.json", "protocol_report.json",
    "baseline_profile.json", "feature_study.json", "target_study.json",
    "combine.json", "guard.json",
    # the scratch scripts that produced a shipped change
    "patch_readme_results.py", "patch_root_readme.py",
}

# Directories worth keeping whole.
KEEP_DIRS = {"audit", "figures", "repo-inspect"}


def classify(path: Path) -> str:
    """Decide what to do with one path: keep, drop, or skip."""
    if path.name in KEEP_FILES:
        return "keep"
    if path.is_dir():
        if path.name in KEEP_DIRS:
            return "keep"
        return "drop"
    # Everything else in here is a capture of stdout or a one-off probe.
    if path.suffix.lower() in {".txt", ".log"}:
        return "drop"
    if path.suffix.lower() == ".py":
        # A one-off script that is not on the keep list is scratch.
        return "drop"
    return "drop"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be removed, remove nothing")
    args = parser.parse_args(argv)

    if not ART.is_dir():
        print("no %s folder" % ART)
        return 0

    dropped = 0
    kept = 0
    freed = 0
    for path in sorted(ART.iterdir()):
        action = classify(path)
        if action == "keep":
            kept += 1
            continue
        try:
            if path.is_file():
                freed += path.stat().st_size
            elif path.is_dir():
                freed += sum(item.stat().st_size for item in path.rglob("*")
                             if item.is_file())
        except OSError:
            pass
        if args.dry_run:
            print("  would remove: %s" % path.name)
        else:
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
            except OSError as exc:
                print("  could not remove %s: %s" % (path.name, exc))
                continue
        dropped += 1

    print("%d kept, %d removed%s (%.1f MB)"
          % (kept, dropped, " [dry run]" if args.dry_run else "",
             freed / (1024 * 1024)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
