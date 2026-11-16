# -*- coding: utf-8 -*-
"""Sync the three-track closed loop, its evidence and the presentation
figures into the AI2 folder.

What this copies
-----------------
``tracks/``       the closed loop that produced the three-track numbers:
                  the stream builder with its judge holdout, the drift
                  detector, the hypothesis bank, the gate, and the report
                  writer.
``runs/``         the run reports, ledgers and hypothesis chains that the
                  loop actually wrote, for all three profiles.
``artifacts/``    the comparison table, the figures, the PPT script and
                  the independent RMSE verification, all generated from
                  those run reports.
``figureuse/``    the presentation figures, regenerated from the same
                  artifacts so a slide cannot show a stale number.

What this does not copy
------------------------
Nothing is copied from the validation set, and no weights are copied:
the competition forbids shipping weights, and the harness regenerates
them at evaluation time.

Run:
    python tools/sync_tracks_into_ai2.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AI2 = ROOT / "AI2"

# The closed loop's own module folder, and the folder the runs wrote.
TRACKS = ("tracks",)
# The generated report artefacts.
ARTIFACT_GLOBS = ("comparison_table.csv", "comparison_table.md", "ppt.md",
                  "figures/*.png", "rmse_verification.json",
                  "rmse_verification.md", "figures/rmse_verification.png",
                  "rul_benchmark.json", "fusion_ablation.json")

# Files that must never travel: the data packages and the weights.
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".bin", ".ckpt", ".pkl",
                       ".joblib", ".safetensors", ".onnx", ".h5"}


def _copy_tree(source: Path, target: Path) -> int:
    """Copy a directory tree, skipping caches and weights."""
    copied = 0
    if not source.is_dir():
        return copied
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            print("  skip (weight): %s" % relative)
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    return copied


def main() -> int:
    if not AI2.is_dir():
        print("AI2 folder not found: %s" % AI2, file=sys.stderr)
        return 1

    print("=== syncing the three-track closed loop into AI2 ===")

    # 1. the code
    for name in TRACKS:
        source = ROOT / name
        if not source.is_dir():
            print("  (no %s/ folder - skipped)" % name)
            continue
        count = _copy_tree(source, AI2 / name)
        print("  %-8s -> AI2/%s  (%d files)" % (name + "/", name, count))

    # The run reports, the summary and the evidence.
    evidence = AI2 / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    reports = 0
    for profile in ("A", "B", "C"):
        source = ROOT / "runs" / profile / "run_report.json"
        if not source.is_file():
            continue
        # The evidence copy is the read-only trail.
        destination = evidence / "tracks" / profile / "run_report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        # The **working** copy matters too: ``tracks.report`` and the
        # integrity tests read ``AI2/runs/<profile>/run_report.json``, so
        # leaving the old numbers there would make the AI2 folder report a
        # score the root no longer produces.  Both copies come from the
        # same file, in the same pass, so they cannot disagree.
        working = AI2 / "runs" / profile / "run_report.json"
        working.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, working)
        reports += 1
    print("  evidence: %d run reports (evidence/ + runs/)" % reports)

    # 3. the summary and the generated artefacts
    summary = ROOT / "runs" / "summary.json"
    if summary.is_file():
        shutil.copy2(summary, evidence / "tracks" / "summary.json")

    artifacts = AI2 / "artifacts"
    copied = 0
    for pattern in ARTIFACT_GLOBS:
        for source in sorted((ROOT / "artifacts").glob(pattern)):
            destination = artifacts / source.relative_to(ROOT / "artifacts")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    print("  artifacts: %d files" % copied)

    # 4. the baselines and the research results
    research = AI2 / "artifacts" / "tracks"
    research.mkdir(parents=True, exist_ok=True)
    for name in ("official_baselines.json", "track_a.json", "track_b.json",
                 "track_c.json", "track_b_final.json", "track_c.sweep.json"):
        source = ROOT / "artifacts" / "tracks" / name
        if source.is_file():
            shutil.copy2(source, research / name)

    # 5. the controlled ablation of the model-side changes.  The README
    #    quotes it, so the numbers have to travel with it.
    ablation = ROOT / "artifacts" / "fusion_ablation.json"
    if ablation.is_file():
        shutil.copy2(ablation, AI2 / "artifacts" / "fusion_ablation.json")

    # 6. the evo submission package's evidence, text-only.  Its model
    #    weights stay behind - the competition forbids shipping them - but
    #    the run summary, the online curve, the audit chain and the
    #    per-version metadata are the proof that the evo numbers in the
    #    README are measurements rather than claims.
    evo_source = ROOT / "artifacts" / "official"
    evo_target = AI2 / "evidence" / "evo"
    if evo_source.is_dir():
        for path in sorted(evo_source.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                continue
            if path.suffix.lower() not in (".json", ".jsonl", ".txt"):
                continue
            relative = path.relative_to(evo_source)
            # The model folder holds one .json beside every .pkl; the .json
            # is the metadata and is worth keeping.
            destination = evo_target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        print("  evidence/evo: the evo run summary, online curve and audit chain")

    # 7. the presentation figures and the overview that goes with them.
    figures = ROOT / "figureuse"
    if figures.is_dir():
        count = _copy_tree(figures, AI2 / "figureuse")
        print("  figureuse -> AI2/figureuse  (%d files)" % count)
    for name in ("智训引擎-系统总思路.md", "基岩智序-完整调试过程.md"):
        overview = ROOT / name
        if overview.is_file():
            shutil.copy2(overview, AI2 / name)
            print("  overview -> AI2/%s" % name)

    # 8. a manifest of what the closed loop actually produced
    manifest = {
        "profiles": {},
        "closed_loop_entrypoint": "python -m tracks.run --all --workspace runs",
        "report_entrypoint": "python -m tracks.report --workspace runs --out artifacts",
        "verification_entrypoint":
            "python -m research.tracks.rmse_verification --out artifacts",
        "figures_entrypoint":
            "python -m research.tracks.make_figures --out figureuse",
    }
    for profile in ("A", "B", "C"):
        path = ROOT / "runs" / profile / "run_report.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest["profiles"][profile] = {
            "package": payload.get("package"),
            "initial_score": payload.get("initial_score"),
            "final_score": payload.get("final_score"),
            "promotions": payload.get("promotions"),
            "recipe": (payload.get("final_refit") or {}).get("recipe"),
        }
    (evidence / "three_track_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  wrote AI2/evidence/three_track_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
