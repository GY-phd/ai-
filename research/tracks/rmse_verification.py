# -*- coding: utf-8 -*-
"""The fused system's RMSE check, across all three tracks.

What this measures
------------------
The closed loop (``tracks/``) produces a champion per profile.  This script
is the *independent check*: it reloads the champion's recipe from the run
report, refits it on the training units only, and scores it on the official
validation set - the same way the official baseline is scored.

The point of doing it here, rather than trusting the run report, is that
the check is written once and applied to all three directions.  A number
that only exists inside one system's own report is a claim; a number that
two independent code paths agree on is a measurement.

For each profile it reports:

``baseline``
    the official static baseline, scored with the competition package's own
    loader and model - never a re-implementation of it;
``system``
    the fused closed-loop champion, refitted from the promoted recipe and
    scored with the same metric;
``per_unit``
    the error broken down by the isolated unit (bearing / engine), which is
    where the remaining error actually lives;
``rmse``
    for the regression profiles, the RMSE on the official metric's scale,
    together with the MAE, so a reviewer can see the error is not being
    carried by a handful of outliers.

Outputs
-------
``artifacts/rmse_verification.json``
    the full machine-readable record;
``artifacts/rmse_verification.md``
    the analysis table;
``artifacts/figures/rmse_verification.png``
    the chart: baseline against system on each track's primary metric,
    plus the per-unit error breakdown for the regression tracks.

Run:
    python -m research.tracks.rmse_verification --out artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tracks.official_baselines import (  # noqa: E402
    DATA_ROOT, PACKAGES, load_checkpoint,
)
from tracks.profiles import get  # noqa: E402
from tracks.run import TrackRunner  # noqa: E402
from tracks.stream import build as build_stream  # noqa: E402


# ------------------------------------------------------------------ helpers
def _unit_of(profile_key, identity):
    """The isolated unit of a row: a bearing or an engine."""
    text = str(identity)
    if profile_key == "B" and "_s" in text:
        return text.rsplit("_s", 1)[0]
    return text


def _order_of(identity):
    """The snapshot index inside a unit."""
    text = str(identity)
    for separator, tail in (("_s", 1), ("#", 1)):
        if separator in text:
            try:
                return int(text.rsplit(separator, 1)[tail])
            except ValueError:
                return 0
    return 0


def _regression_metrics(y_true, y_pred) -> dict:
    """RMSE and friends on the official metric's scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    total = max(float(np.sum((y_true - y_true.mean()) ** 2)), 1e-12)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - float(np.sum(error ** 2)) / total),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _macro_f1(y_true, predicted) -> float:
    """Macro-averaged F1 over the classes that appear in the truth."""
    classes = sorted(set(int(v) for v in y_true))
    scores = []
    for cls in classes:
        true_positive = int(np.sum((predicted == cls) & (y_true == cls)))
        false_positive = int(np.sum((predicted == cls) & (y_true != cls)))
        false_negative = int(np.sum((predicted != cls) & (y_true == cls)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def _classification_metrics(y_true, probability) -> dict:
    """Accuracy, macro-F1 and the error distribution."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability.argmax(axis=1) if probability.ndim > 1
                 else (probability >= 0.5).astype(int))
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "macro_f1": _macro_f1(y_true, predicted),
        "max_abs_error": float(np.max(np.abs(probability.max(axis=1) - 1.0))),
        "bias": 0.0,
    }


def _per_unit(y_true, y_pred, units) -> dict:
    """The error broken down by the isolated unit."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for unit in sorted(set(units)):
        mask = np.asarray([u == unit for u in units])
        if not mask.any():
            continue
        error = y_pred[mask] - y_true[mask]
        out[str(unit)] = {
            "rmse": round(float(np.sqrt(np.mean(error ** 2))), 3),
            "mae": round(float(np.mean(np.abs(error))), 3),
            "rows": int(mask.sum()),
        }
    return out


# ------------------------------------------------------------------ verify
def verify_profile(profile_key: str) -> dict:
    """Score the official baseline and the fused champion on one track."""
    profile = get(profile_key)
    report_path = ROOT / "runs" / profile_key / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recipe = (report.get("final_refit") or {}).get("recipe") or {}
    if not recipe:
        raise RuntimeError("profile %s has no promoted recipe" % profile_key)

    # Rebuild the **same** split the closed loop used, so the champion is
    # refitted on exactly the rows it was shipped with: the initial part,
    # the stream and the judge set.  Reading the official ``train.jsonl``
    # instead would silently drop the stream and judge rows, and the check
    # would then measure a different model than the one that ships.
    #
    # The split is recovered from the unit names the run recorded rather
    # than from the parameters that produced it, so the check cannot drift
    # away from the run if the defaults change.
    stream_meta = report.get("stream_meta") or {}
    judge_units = set(stream_meta.get("judge_units") or [])
    stream_units = set(stream_meta.get("stream_units") or [])
    if not judge_units and not stream_units:
        raise RuntimeError("profile %s does not record its split" % profile_key)

    from tracks.stream import build as build_stream

    # Try the split parameters the run actually used, and accept the one
    # whose unit assignment matches what the run recorded.
    candidate_params = [
        (int(stream_meta.get("holdout_units", 0) or 0), len(judge_units)),
        (4, 2), (3, 1), (4, 1), (3, 2),
    ]
    initial = stream = judge = validation = None
    # Profile A does not hold out whole units: its judge set is a
    # class-stratified slice of the validation batch, so the unit names
    # cannot be matched against the split.  For A the split is fixed by
    # the profile, so the first attempt is the right one.
    within_batch = bool(stream_meta.get("mode") == "within_batch_stream")
    for n_holdout, n_judge in candidate_params:
        if n_holdout <= 0 or n_judge < 0:
            continue
        try:
            built = build_stream(profile_key, n_holdout, n_judge)
        except Exception:
            continue
        candidate_initial, candidate_stream, candidate_judge, candidate_validation, _ = built
        if within_batch:
            initial, stream, judge, validation = (candidate_initial, candidate_stream,
                                                  candidate_judge, candidate_validation)
            break
        got_judge = set(np.asarray(candidate_judge["groups"]).tolist())
        got_stream = set(np.asarray(candidate_stream["groups"]).tolist())
        if got_judge == judge_units and got_stream == stream_units:
            initial, stream, judge, validation = (candidate_initial, candidate_stream,
                                                  candidate_judge, candidate_validation)
            break
    if initial is None:
        raise RuntimeError(
            "profile %s: could not reproduce the recorded split "
            "(judge=%d units, stream=%d units)"
            % (profile_key, len(judge_units), len(stream_units)))

    pieces = [piece for piece in (initial, stream, judge)
              if piece is not None and len(piece["y"])]
    train_x = np.concatenate([piece["x"] for piece in pieces])
    train_y = np.concatenate([piece["y"] for piece in pieces])
    train_groups = np.concatenate([piece["groups"] for piece in pieces])
    train_order = np.concatenate([piece["order"] for piece in pieces])
    train_paths = initial["paths"]

    runner = TrackRunner(profile_key, ROOT / "runs", {"model": {}})
    model, fit_report = runner.fit(
        train_x, train_y, train_paths, train_groups, train_order,
        "verification", model_overrides=recipe)

    validation_groups = np.asarray(validation["groups"])
    validation_order = np.asarray(validation["order"], dtype=float)
    system_pred = np.asarray(
        runner.predict(model, validation["x"], validation_groups, validation_order),
        dtype=float)
    # -- the official baseline, through the competition package's own code
    _, official_model = load_checkpoint(profile_key)
    official_pred = np.asarray(official_model.predict(validation["x"]))

    result = {
        "profile": profile_key,
        "task": profile.task,
        "package": PACKAGES[profile_key],
        "validation_rows": int(len(validation["y"])),
        "recipe": recipe,
        "refit_rows": int(fit_report["rows"]),
        "refit_parameter_count": int(fit_report["parameter_count"]),
    }

    if profile.task == "classification":
        baseline = _classification_metrics(validation["y"], official_pred)
        system = _classification_metrics(validation["y"], system_pred)
        result["metric"] = "macro_f1"
        result["baseline_primary"] = round(baseline["macro_f1"], 4)
        result["system_primary"] = round(system["macro_f1"], 4)
        result["primary_gain_pct"] = round(
            (system["macro_f1"] - baseline["macro_f1"]) / baseline["macro_f1"] * 100, 2)
        result["rmse_gain_pct"] = None
        # The classification track has no RMSE; the per-unit breakdown is
        # reported on the error rate instead.
        for name, truth, prediction in (("baseline", validation["y"], official_pred),
                                        ("system", validation["y"], system_pred)):
            block = {}
            for unit in sorted(set(validation_groups)):
                mask = validation_groups == unit
                predicted = (prediction.argmax(axis=1) if np.ndim(prediction) > 1
                             else (prediction >= 0.5).astype(int))
                truth_i = np.asarray(truth).astype(int)[mask]
                block[str(unit)] = {
                    "accuracy": round(float(np.mean(predicted[mask] == truth_i)), 4),
                    "macro_f1": round(_macro_f1(truth_i, predicted[mask]), 4),
                    "rows": int(mask.sum()),
                }
            result["per_unit_%s" % name] = block
    else:
        baseline = _regression_metrics(validation["y"], official_pred)
        system = _regression_metrics(validation["y"], system_pred)
        result["metric"] = "rmse"
        result["baseline_primary"] = round(baseline["rmse"], 4)
        result["system_primary"] = round(system["rmse"], 4)
        result["rmse_gain_pct"] = round(
            (baseline["rmse"] - system["rmse"]) / baseline["rmse"] * 100, 2)
        result["primary_gain_pct"] = result["rmse_gain_pct"]
        result["per_unit_baseline"] = _per_unit(
            validation["y"], official_pred, validation_groups)
        result["per_unit_system"] = _per_unit(
            validation["y"], system_pred, validation_groups)

    result["baseline"] = baseline
    result["system"] = system
    return result


# ------------------------------------------------------------------ output
def write_markdown(results: list, path: Path) -> None:
    """The analysis table."""
    lines = [
        "# 融合系统 RMSE 检验",
        "",
        "> 由 `python -m research.tracks.rmse_verification` 生成。"
        "静态基线直接调用赛题包的 loader 与模型；系统分由闭环产出的配方"
        "在同一份数据上重新拟合后打分——**两条独立的代码路径，同一个口径**。",
        "",
        "## 1. 三方向总表",
        "",
        "| 方向 | 任务 | 指标 | 静态基线 | 智训引擎 | 相对变化 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        if result["metric"] == "rmse":
            lines.append("| %s | 回归 | RMSE | %.3f | **%.3f** | %+.2f%% |" % (
                result["profile"], result["baseline_primary"],
                result["system_primary"], result["rmse_gain_pct"]))
        else:
            lines.append("| %s | 分类 | Macro-F1 | %.4f | **%.4f** | %+.2f%% |" % (
                result["profile"], result["baseline_primary"],
                result["system_primary"], result["primary_gain_pct"]))
    lines += [
        "",
        "## 2. 回归方向的误差明细",
        "",
        "| 方向 | 系统 | RMSE | MAE | R² | 偏差 | 最大单点误差 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if result["task"] != "regression":
            continue
        for label, key in (("静态基线", "baseline"), ("智训引擎", "system")):
            block = result[key]
            lines.append("| %s | %s | %.3f | %.3f | %.3f | %+.3f | %.3f |" % (
                result["profile"], label, block["rmse"], block["mae"], block["r2"],
                block["bias"], block["max_abs_error"]))
    lines += ["", "## 3. 逐单元误差（回归方向）", ""]
    for result in results:
        if result["task"] != "regression":
            continue
        lines += [
            "### %s 方向" % result["profile"],
            "",
            "| 单元 | 样本数 | 静态基线 RMSE | 智训引擎 RMSE | 相对变化 |",
            "|---|---:|---:|---:|---:|",
        ]
        for unit, system_block in result["per_unit_system"].items():
            baseline_block = result["per_unit_baseline"].get(unit, {})
            base_rmse = baseline_block.get("rmse", float("nan"))
            gain = ((base_rmse - system_block["rmse"]) / base_rmse * 100
                    if base_rmse else 0.0)
            lines.append("| %s | %d | %.3f | **%.3f** | %+.2f%% |" % (
                unit, system_block["rows"], base_rmse, system_block["rmse"], gain))
        lines.append("")

    lines += [
        "## 4. 分类方向的误差明细",
        "",
        "| 方向 | 系统 | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
    ]
    for result in results:
        if result["task"] != "classification":
            continue
        for label, key in (("静态基线", "baseline"), ("智训引擎", "system")):
            block = result[key]
            lines.append("| %s | %s | %.4f | **%.4f** |" % (
                result["profile"], label, block["accuracy"], block["macro_f1"]))
        lines += ["", "### %s 方向逐批次明细" % result["profile"], "",
                  "| 批次 | 样本数 | 静态基线 Accuracy | 智训引擎 Accuracy |",
                  "|---|---:|---:|---:|"]
        for unit, system_block in result["per_unit_system"].items():
            baseline_block = result["per_unit_baseline"].get(unit, {})
            lines.append("| %s | %d | %.4f | **%.4f** |" % (
                unit, system_block["rows"], baseline_block.get("accuracy", float("nan")),
                system_block["accuracy"]))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(results: list, path: Path) -> bool:
    """The chart: baseline against system, plus the per-unit breakdown."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:  # pragma: no cover - plotting is optional
        print("  (skipping the figure: %s)" % exc)
        return False

    # Pick a font that has the Chinese glyphs; otherwise the labels render
    # as empty boxes.  Same preference list as ``research.tracks.make_figures``.
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in preferred if name in available]
    plt.rcParams["font.sans-serif"] = (chosen + ["DejaVu Sans"]) if chosen else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    regressions = [r for r in results if r["task"] == "regression"]
    classifications = [r for r in results if r["task"] == "classification"]
    columns = 2 + len(regressions)
    fig, axes = plt.subplots(1, columns, figsize=(5.0 * columns, 4.6))
    axes = [axes] if columns == 1 else list(axes)

    # Panel 1: RMSE on each regression track.
    ax = axes[0]
    if regressions:
        labels = [r["profile"] for r in regressions]
        baseline = [r["baseline"]["rmse"] for r in regressions]
        system = [r["system"]["rmse"] for r in regressions]
        positions = np.arange(len(labels))
        width = 0.36
        ax.bar(positions - width / 2, baseline, width, label="静态基线", color="#8c8c8c")
        ax.bar(positions + width / 2, system, width, label="智训引擎", color="#2f6f9f")
        for index, (base, sys_) in enumerate(zip(baseline, system)):
            ax.text(index - width / 2, base, "%.1f" % base, ha="center", va="bottom",
                    fontsize=9)
            ax.text(index + width / 2, sys_, "%.1f" % sys_, ha="center", va="bottom",
                    fontsize=9, color="#2f6f9f")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE (越小越好)")
        ax.set_title("回归方向 RMSE 检验")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")

    # Panel 2: the primary metric on every track.
    ax = axes[1]
    labels, baseline, system = [], [], []
    for result in results:
        if result["metric"] == "rmse":
            labels.append("%s\nRMSE" % result["profile"])
            baseline.append(result["baseline"]["rmse"])
            system.append(result["system"]["rmse"])
        else:
            labels.append("%s\nMacro-F1" % result["profile"])
            baseline.append(result["baseline"]["macro_f1"])
            system.append(result["system"]["macro_f1"])
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, baseline, width, label="静态基线", color="#8c8c8c")
    ax.bar(positions + width / 2, system, width, label="智训引擎", color="#2f6f9f")
    for index, (base, sys_) in enumerate(zip(baseline, system)):
        ax.text(index - width / 2, base, "%.4g" % base, ha="center", va="bottom", fontsize=9)
        ax.text(index + width / 2, sys_, "%.4g" % sys_, ha="center", va="bottom",
                fontsize=9, color="#2f6f9f")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("各方向主指标")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3+: the per-unit error, where the remaining error lives.
    for offset, result in enumerate(regressions):
        ax = axes[2 + offset]
        units = list(result["per_unit_system"].keys())
        base_values = [result["per_unit_baseline"].get(u, {}).get("rmse", 0.0)
                       for u in units]
        system_values = [result["per_unit_system"][u]["rmse"] for u in units]
        positions = np.arange(len(units))
        width = 0.36
        ax.bar(positions - width / 2, base_values, width, label="静态基线",
               color="#8c8c8c")
        ax.bar(positions + width / 2, system_values, width, label="智训引擎",
               color="#2f6f9f")
        ax.set_xticks(positions)
        ax.set_xticklabels([u.replace("xjtu_", "") for u in units], fontsize=8,
                           rotation=20)
        ax.set_ylabel("RMSE")
        ax.set_title("%s 方向逐单元 RMSE" % result["profile"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("融合系统 RMSE 检验（静态基线 vs 智训引擎，同口径独立复算）",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--profiles", default="A,B,C")
    args = parser.parse_args(argv)

    out = Path(args.out)
    results = []
    for profile_key in [p.strip() for p in args.profiles.split(",") if p.strip()]:
        print("=== profile %s ===" % profile_key)
        try:
            result = verify_profile(profile_key)
        except Exception as exc:
            print("  (skipped: %s: %s)" % (type(exc).__name__, exc))
            continue
        results.append(result)
        if result["metric"] == "rmse":
            print("  RMSE      baseline %8.3f  ->  system %8.3f   (%+.2f%%)"
                  % (result["baseline"]["rmse"], result["system"]["rmse"],
                     result["rmse_gain_pct"]))
            print("  MAE       baseline %8.3f  ->  system %8.3f"
                  % (result["baseline"]["mae"], result["system"]["mae"]))
        else:
            print("  Macro-F1  baseline %8.4f  ->  system %8.4f   (%+.2f%%)"
                  % (result["baseline"]["macro_f1"], result["system"]["macro_f1"],
                     result["primary_gain_pct"]))
            print("  Accuracy  baseline %8.4f  ->  system %8.4f"
                  % (result["baseline"]["accuracy"], result["system"]["accuracy"]))

    if not results:
        print("no profile could be verified")
        return 1

    payload = {
        "protocol": "the official static baseline is scored with the "
                    "competition package's own loader and model; the fused "
                    "champion is refitted from the recipe the closed loop "
                    "promoted and scored with the same metric",
        "data_root": str(DATA_ROOT),
        "results": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rmse_verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  wrote %s" % (out / "rmse_verification.json"))

    write_markdown(results, out / "rmse_verification.md")
    print("  wrote %s" % (out / "rmse_verification.md"))

    if write_figure(results, out / "figures" / "rmse_verification.png"):
        print("  wrote %s" % (out / "figures" / "rmse_verification.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
