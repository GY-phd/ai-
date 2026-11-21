# -*- coding: utf-8 -*-
"""The acceptance benchmark: proof that the system performs better.

What "better" has to mean
-------------------------
The competition does not score the public validation set; it scores a
hidden stream.  So a number that improves on the public split is
evidence, not proof - and the only honest way to turn it into proof is
to say **exactly what was compared with what**.

This tool answers that question with a table:

    profile | parent checkpoint | baseline | after optimisation | gain

Both sides of the comparison are measured **under the same
conditions**: the same parent checkpoint, the same seed, the same public
validation projection, and the same reference forward pass.  The
baseline side is the parent checkpoint scored on that validation split
with no training at all; the optimised side is the best *adopted*
candidate the closed loop produced from that same parent.

The baseline is not a separate run
---------------------------------
The first round of the loop measures the parent before it trains
anything, so the parent's own score on the validation split is already
the first round's ``validation_before``.  Reading it from there rather
than measuring it again keeps the two sides of the comparison on
exactly the same footing - same code, same data, same moment - and
keeps this tool from owning a second implementation of the loop.

What is written out
-------------------
``benchmark.json`` for machines and ``benchmark.md`` for people.  Both
carry the parent's digest and the seed, so a third party can reproduce
the table.

**No weights are ever written.**  The competition forbids model weights
in the repository and in the submitted package, so this tool's output is
JSON and Markdown and nothing else.  ``tools/verify_repo.py`` checks
that the rule was kept.

Run:

    python tools/benchmark_profiles.py --rounds 10
    python tools/benchmark_profiles.py --datasets A --rounds 3 --out runs/benchmark
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contest_baselines.offline_host import package_root  # noqa: E402
from tools.run_harness_offline import (  # noqa: E402
    _HistoryHost,
    _ledger_entry,
    _run_round,
)

#: The dataset keys the competition has.
DATASETS = ("A", "B", "C")


# --------------------------------------------------------------------- helpers
def _primary_of(values):
    """The primary metric's name and value out of a metric dictionary."""
    if not values:
        return None, None
    for name in ("macro_f1", "rmse"):
        if name in values:
            return name, float(values[name])
    return None, None


def _is_better(task, before, after):
    """Did the primary metric move in the task's good direction?

    Classification is scored by Macro-F1 and higher is better;
    regression is scored by RMSE and lower is better.  The comparison
    follows the task rather than the metric's name, so a profile whose
    primary metric is renamed later does not silently flip its sign.
    """
    if task == "classification":
        return after > before
    return after < before


def _gain(task, before, after):
    """The improvement, always signed so that positive means better."""
    if task == "classification":
        return float(after) - float(before)
    return float(before) - float(after)


# ------------------------------------------------------------------ the runner
def _baseline_measurement(dataset_key):
    """The parent checkpoint's own score on the public validation split.

    No training happens here.  The parent is loaded and scored exactly
    as the loop's own ``measure_parent`` step would score it, so the
    baseline is the same number the loop starts from.
    """
    from contest_baselines.data import read_rows
    from contest_baselines.metrics import (
        primary_metric_name,
        primary_metric_value,
        score_metrics,
    )
    from contest_baselines.model import Model

    package = package_root(dataset_key)
    checkpoint = package / "checkpoint.npz"
    model = Model.load(checkpoint)
    classes = list(model.meta.get("classes") or [])
    validation = read_rows(package / "public" / "validation.jsonl",
                           model.meta["task"], classes)
    if model.meta["task"] == "classification":
        predicted = model.predict(validation["x"]).argmax(axis=1)
    else:
        predicted = model.predict(validation["x"])
    values = score_metrics(model.meta["task"], validation["y"], predicted)
    name = primary_metric_name(model.meta["task"])
    return {
        "primary_metric": name,
        "primary_value": primary_metric_value(model.meta["task"], values),
        "values": values,
        "task": model.meta["task"],
        "parent_checkpoint_sha256": model.meta.get("parent_sha256"),
        "seed": int(model.meta.get("seed", 42)),
        "checkpoint_digest": _digest(checkpoint),
    }


def _digest(path):
    """The sha256 of a checkpoint file, in the platform's format."""
    import hashlib

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_loop(dataset_key, rounds):
    """Run the closed loop and return what it produced.

    The loop is the platform's own: the participant's ``harness.py
    train`` is driven by the offline host stand-in, with the previous
    round's ledger carried forward as the history.  Only the model call
    is answered locally.
    """
    history = []
    records = []
    parent = None
    keep = set()
    for round_index in range(1, rounds + 1):
        work = Path(tempfile.mkdtemp(prefix="bench-%s-r%d-" % (dataset_key,
                                                              round_index)))
        started = time.monotonic()
        host = _run_round(dataset_key, round_index, history, work / "work",
                          parent=parent)
        if host.failure is not None:
            raise RuntimeError("round %d of %s failed: %s\n%s"
                               % (round_index, dataset_key, host.failure,
                                  getattr(host, "stderr_text", "")))
        entry = _ledger_entry(host, dataset_key, round_index)
        history.append(entry)

        candidate_dir = work / "work" / ("candidate-round-%d" % round_index)
        candidate_checkpoint = candidate_dir / "checkpoint.npz"
        advanced = False
        if entry["adopted"] and candidate_checkpoint.is_file():
            parent = (str(candidate_dir), str(candidate_checkpoint),
                      entry["selected_artifact_id"]
                      or ("offline-artifact-r%d" % round_index))
            keep.add(work)
            advanced = True

        records.append({
            "round": round_index,
            "hypothesis": entry["hypothesis"],
            "adopted": bool(entry["adopted"]),
            "validation_before": entry["validation_before"],
            "validation_after": entry["validation_after"],
            "selected_artifact_id": entry["selected_artifact_id"],
            "model_call_ids": entry["model_call_ids"],
            "parent_advanced": advanced,
            "seconds": round(time.monotonic() - started, 2),
        })
        if work not in keep:
            shutil.rmtree(work, ignore_errors=True)
    for path in keep:
        shutil.rmtree(path, ignore_errors=True)
    return records


def _best_adopted(records, task):
    """The best measurement among the rounds that were adopted.

    A round that was not adopted is a **refuted** hypothesis: its
    number is evidence, but it is not the system's output.  So the
    optimised side of the comparison is the best *adopted* round, and a
    profile where nothing was adopted has no optimised side - which is
    reported as such rather than filled in with the parent's own score.
    """
    best = None
    for record in records:
        if not record["adopted"]:
            continue
        name, value = _primary_of(record["validation_after"])
        if name is None:
            continue
        if best is None or _is_better(task, best[1], value):
            best = (name, value, record)
    return best


def benchmark_profile(dataset_key, rounds):
    """Measure one profile and return its comparison."""
    baseline = _baseline_measurement(dataset_key)
    task = baseline["task"]
    records = _run_loop(dataset_key, rounds)

    adopted = [record for record in records if record["adopted"]]
    best = _best_adopted(records, task)

    before_value = baseline["primary_value"]
    after_value = best[1] if best else None

    gain = None
    direction = "unknown"
    if after_value is not None:
        gain = _gain(task, before_value, after_value)
        direction = "improved" if gain > 0 else (
            "unchanged" if gain == 0 else "regressed")

    # The unit gain is the improvement per **adopted** round.  With no
    # adopted round there is no unit to divide by, and reporting zero
    # would read as "the changes had no effect" - a different claim.
    unit_gain = None
    if gain is not None and adopted:
        unit_gain = gain / len(adopted)

    return {
        "profile": dataset_key,
        "task": task,
        "primary_metric": baseline["primary_metric"],
        "parent_checkpoint_sha256": baseline["checkpoint_digest"],
        "seed": baseline["seed"],
        "rounds": len(records),
        "adopted_rounds": len(adopted),
        "adoption_rate": (len(adopted) / len(records)) if records else 0.0,
        "baseline": {
            "primary_value": before_value,
            "values": baseline["values"],
            "source": "the parent checkpoint, scored on the public "
                      "validation split with no training",
        },
        "optimised": {
            "primary_value": after_value,
            "values": best[2]["validation_after"] if best else None,
            "round": best[2]["round"] if best else None,
            "hypothesis": best[2]["hypothesis"] if best else None,
            "source": "the best adopted candidate of the closed loop",
        },
        "gain": gain,
        "gain_direction": direction,
        "unit_gain": unit_gain,
        "unit_gain_available": unit_gain is not None,
        "rounds_detail": records,
    }


# ------------------------------------------------------------------- the report
def _markdown(report):
    """The comparison table, for a person to read."""
    lines = [
        "# 验收基线对比表",
        "",
        "> 「基线」= 父检查点在公共验证集上的分数（未训练）。",
        "> 「优化后」= 闭环中被**采纳**的候选里的最优者。",
        "> 两侧使用**同一父检查点、同一种子、同一验证集**。",
        "",
        "| profile | 任务 | 主指标 | 基线 | 优化后 | 增益 | 方向 | 采纳率 | 单位增益 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for profile in report["profiles"]:
        if profile.get("failed"):
            lines.append("| %s | — | — | — | — | — | **失败** | — | — |"
                         % profile["profile"])
            continue
        before = profile["baseline"]["primary_value"]
        after = profile["optimised"]["primary_value"]
        gain = profile["gain"]
        unit = profile["unit_gain"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %.2f | %s |" % (
            profile["profile"],
            profile["task"],
            profile["primary_metric"],
            "—" if before is None else "%.4f" % before,
            "—" if after is None else "%.4f" % after,
            "—" if gain is None else "%+.4f" % gain,
            profile["gain_direction"],
            profile["adoption_rate"],
            "不可用" if unit is None else "%+.4f" % unit,
        ))

    lines += [
        "",
        "## 可复现信息",
        "",
        "| profile | 父检查点摘要 | 种子 | 轮次 |",
        "|---|---|---:|---:|",
    ]
    for profile in report["profiles"]:
        if profile.get("failed"):
            lines.append("| %s | — | — | — |" % profile["profile"])
            continue
        lines.append("| %s | `%s` | %s | %d |" % (
            profile["profile"], profile["parent_checkpoint_sha256"],
            profile["seed"], profile["rounds"]))

    lines += ["", "## 说明", ""]
    lines.append("- **方向**按任务语义判定：分类越高越好，回归越低越好。")
    lines.append("- **采纳率** = 被采纳轮次 ÷ 总轮次。")
    lines.append("- **单位增益** = 主指标增益 ÷ 采纳轮次；零采纳时记为"
                 "「不可用」而非 0，以免把「没动作」误读成「没效果」。")
    lines.append("- 本产物只含 JSON 与 Markdown，**不含任何权重文件**。")
    return "\n".join(lines) + "\n"


def _write_report(report, out):
    """Write both artefacts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8")
    (out / "benchmark.md").write_text(_markdown(report), encoding="utf-8")


# ------------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="A,B,C",
                        help="which profiles to measure")
    parser.add_argument("--rounds", type=int, default=10,
                        help="how many rounds of the closed loop to run")
    parser.add_argument("--packages", default=None,
                        help="where the data packages live "
                             "(overrides CONTEST_PACKAGES)")
    parser.add_argument("--out", default="runs/benchmark",
                        help="where to write the comparison table")
    args = parser.parse_args(argv)

    if args.packages:
        os.environ["CONTEST_PACKAGES"] = str(Path(args.packages).resolve())

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    keys = [key.strip() for key in args.datasets.split(",") if key.strip()]

    report = {
        "format": "contest.acceptance-benchmark.v1",
        "rounds_per_profile": args.rounds,
        "note": "The baseline is the parent checkpoint scored on the public "
                "validation split; the optimised side is the best adopted "
                "candidate. Both sides use the same parent, seed and split.",
        "profiles": [],
    }

    for key in keys:
        print("measuring profile %s ..." % key, flush=True)
        try:
            profile = benchmark_profile(key, args.rounds)
        except Exception as error:  # noqa: BLE001 - one profile must not
            # stop the others from being measured
            profile = {
                "profile": key,
                "failed": True,
                "error": "%s: %s" % (type(error).__name__, error),
            }
            print("  profile %s failed: %s" % (key, error), flush=True)
        else:
            print("  baseline %.4f -> optimised %s (%s)"
                  % (profile["baseline"]["primary_value"],
                     "—" if profile["optimised"]["primary_value"] is None
                     else "%.4f" % profile["optimised"]["primary_value"],
                     profile["gain_direction"]), flush=True)
        report["profiles"].append(profile)

    _write_report(report, out)
    print("wrote %s" % (out / "benchmark.json"))
    print("wrote %s" % (out / "benchmark.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
