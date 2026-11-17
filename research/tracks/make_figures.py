# -*- coding: utf-8 -*-
"""The figures for the presentation, in one place.

Every figure here is generated from the artifacts - the run reports, the
RMSE verification, the online curve - so a slide cannot show a number
that the repository does not contain.  The labels are Chinese because the
slides are, and the figures are read from across a room.

Outputs (all under ``--out``, default ``figureuse``):

``01_三方向主指标对比.png``
    the headline: official baseline against the fused system on each
    track's primary metric, with the gain labelled;
``02_RMSE检验_回归方向.png``
    the RMSE check on the two regression tracks, baseline against system,
    plus the per-unit breakdown where the remaining error lives;
``03_分类方向_逐批次混淆与准确率.png``
    the gas-classification track: per-batch accuracy for both systems;
``04_闭环决策账本.png``
    the ledger of the closed loop: every cycle's decision, per track;
``05_在线自适应曲线.png``
    the evo package's online curve - the static baseline degrading while
    the adaptive system tracks ahead;
``06_逐单元误差分解.png``
    the per-unit error on the regression tracks, side by side.

Run:
    python -m research.tracks.make_figures --out figureuse
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

# A readable palette that survives a projector: the baseline is a flat
# grey, the system is a saturated blue, and the accent is warm.
GREY = "#9aa0a6"
BLUE = "#1f6fb2"
LIGHT = "#7fb3dc"
ACCENT = "#d1495b"
GREEN = "#2e8b57"

TRACK_NAMES = {
    "A": "A 方向\n气体分类",
    "B": "B 方向\n轴承 RUL",
    "C": "C 方向\n发动机 RUL",
}


def _style():
    """Make the plots presentable without a style package."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required: %s" % exc)

    # Pick a font that has the Chinese glyphs.  On Windows that is
    # Microsoft YaHei; elsewhere the DejaVu fallback keeps the English
    # labels readable and the Chinese as boxes, which is still better than
    # a crash.
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in preferred if name in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    return plt


def _read(path: Path):
    """Read a JSON artifact, or return None."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ------------------------------------------------------------------ figures
def figure_headline(plt, out: Path, verification: dict, table: list) -> Path:
    """The headline comparison: official baseline against the fused system."""
    results = verification.get("results", []) if verification else []
    if not results:
        return None

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for index, result in enumerate(results):
        ax = axes[index]
        profile = result["profile"]
        is_regression = result["metric"] == "rmse"

        if is_regression:
            base = result["baseline"]["rmse"]
            system = result["system"]["rmse"]
            labels = ["静态基线", "智训引擎"]
            values = [base, system]
            colors = [GREY, BLUE]
            unit = "分钟" if profile == "B" else "循环"
            ax.set_ylabel("RMSE（%s）" % unit)
            gain = result["rmse_gain_pct"]
            fmt = "%.1f" % base
            fmt2 = "%.1f" % system
        else:
            base = result["baseline"]["macro_f1"]
            system = result["system"]["macro_f1"]
            labels = ["静态基线", "智训引擎"]
            values = [base, system]
            colors = [GREY, BLUE]
            ax.set_ylabel("Macro-F1")
            gain = result["primary_gain_pct"]
            fmt = "%.4f" % base
            fmt2 = "%.4f" % system

        bars = ax.bar(labels, values, color=colors, width=0.55)
        for bar, value, text in zip(bars, values, (fmt, fmt2)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    text, ha="center", va="bottom", fontsize=12, fontweight="bold")

        # The gain, as the single most important number on the slide.
        ax.set_title("%s\n%s" % (TRACK_NAMES.get(profile, profile),
                                 "RMSE 降低 %.2f%%" % gain if is_regression
                                 else "Macro-F1 提升 %.2f%%" % gain))
        ax.set_ylim(0, max(values) * 1.30)
        ax.tick_params(axis="x", labelsize=10)

    figure.suptitle("融合系统 vs 静态基线：三个赛题方向的主指标（同口径独立复算）",
                     fontsize=15, y=1.04)
    figure.tight_layout()
    path = out / "01_三方向主指标对比.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_rmse(plt, out: Path, verification: dict) -> Path:
    """The RMSE check on the regression tracks, plus the per-unit breakdown."""
    results = [r for r in (verification or {}).get("results", [])
               if r["metric"] == "rmse"]
    if not results:
        return None

    figure, axes = plt.subplots(1, len(results) + 1, figsize=(6.0 * (len(results) + 1), 5.0))
    axes = np.atleast_1d(axes)

    # Panel 1: RMSE, baseline against system, per regression track.
    ax = axes[0]
    labels, base_values, system_values = [], [], []
    for result in results:
        unit = "分钟" if result["profile"] == "B" else "循环"
        labels.append("%s\nRMSE/%s" % (result["profile"], unit))
        base_values.append(result["baseline"]["rmse"])
        system_values.append(result["system"]["rmse"])
    positions = np.arange(len(labels))
    width = 0.34
    ax.bar(positions - width / 2, base_values, width, label="静态基线", color=GREY)
    ax.bar(positions + width / 2, system_values, width, label="智训引擎", color=BLUE)
    for index, (base, system) in enumerate(zip(base_values, system_values)):
        ax.text(index - width / 2, base, "%.1f" % base, ha="center", va="bottom",
                fontsize=10)
        ax.text(index + width / 2, system, "%.1f" % system, ha="center", va="bottom",
                fontsize=10, color=BLUE, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("RMSE（越小越好）")
    ax.set_title("回归方向 RMSE 检验")
    ax.legend()
    ax.set_ylim(0, max(base_values) * 1.25)

    # Panel 2+: the per-unit error, where the remaining error actually lives.
    #
    # The unit labels are the problem: profile C's validation set holds
    # 111 engines, and printing one label per engine turns the axis into
    # a grey smear.  The fix is to group by the sub-dataset (FD001 ...
    # FD004), label the *group* on the axis, and let the bar chart carry
    # the per-engine detail.  A reader can still ask "how did FD004 do?"
    # without squinting at 27 overlapping strings.
    for offset, result in enumerate(results):
        ax = axes[1 + offset]
        units = list(result["per_unit_system"].keys())
        base_unit = [result["per_unit_baseline"].get(u, {}).get("rmse", 0.0)
                     for u in units]
        system_unit = [result["per_unit_system"][u]["rmse"] for u in units]
        positions = np.arange(len(units))
        width = 0.36
        ax.bar(positions - width / 2, base_unit, width, label="静态基线", color=GREY)
        ax.bar(positions + width / 2, system_unit, width, label="智训引擎", color=BLUE)

        # Group the units by their sub-dataset and label the groups.
        groups = {}
        for unit in units:
            group = unit.split("#", 1)[0] if "#" in unit else (
                unit.split("_")[0] if "_" in unit else unit)
            groups.setdefault(group, []).append(unit)
        labels = []
        for group, members in groups.items():
            labels.extend([group] + [""] * (len(groups[group]) - 1))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9, rotation=0)
        # Mark where each sub-dataset's block ends.
        cursor = 0
        for group, members in groups.items():
            cursor += len(members)
            if cursor < len(units):
                ax.axvline(cursor - 0.5, color="#cccccc", linewidth=0.9, zorder=0)
        ax.set_ylabel("RMSE")
        ax.set_title("%s 方向逐单元 RMSE（按%s分组）"
                     % (result["profile"],
                        "子数据集" if "#" in units[0] else "轴承"))
        ax.legend()
        ax.margins(x=0.01)

    figure.suptitle("融合系统 RMSE 检验：官方基线 vs 智训引擎，同口径独立复算",
                     fontsize=15, y=1.03)
    figure.tight_layout()
    path = out / "02_RMSE检验_回归方向.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_classification(plt, out: Path, verification: dict, reports: dict) -> Path:
    """The gas-classification track: the progression and the class breakdown.

    A's validation set is a single batch, so a per-batch chart would be
    one bar.  What there is to show is the *progression*: what the score
    was at each cycle, and how the final model compares with the
    baseline on the metrics that matter for a six-class problem.
    """
    results = [r for r in (verification or {}).get("results", [])
               if r["task"] == "classification"]
    if not results:
        return None
    result = results[0]

    figure, axes = plt.subplots(1, 2, figsize=(13.8, 4.9))

    # Panel 1: the score at each closed-loop cycle.
    ax = axes[0]
    report = (reports or {}).get(result["profile"], {})
    ledger = report.get("ledger") or []
    cycles = []
    scores = []
    for entry in ledger:
        champion = entry.get("champion_score_on_same_slice") or {}
        if not isinstance(champion, dict):
            continue
        # The classification profile's slice scores use ``f1``/``acc``;
        # the regression ones use ``rmse``/``mae``.  Both spellings appear
        # in the ledger, so look for the classification ones by name.
        value = champion.get("f1")
        if value is None:
            value = champion.get("macro_f1")
        if value is None:
            continue
        cycles.append(int(entry.get("cycle", len(cycles))) + 1)
        scores.append(float(value))
    if cycles:
        positions = np.arange(len(cycles))
        ax.plot(positions, scores, marker="o", linewidth=2.6, color=BLUE,
                label="闭环每轮冠军得分")
        ax.scatter(positions, scores, s=90, color=BLUE, zorder=3)
        for position, value in zip(positions, scores):
            ax.text(position, value, "%.4f" % value, ha="center", va="bottom",
                    fontsize=9.5, color=BLUE, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(["第 %d 轮" % c for c in cycles], fontsize=9.5)
        ax.set_ylim(min(scores) * 0.94, max(scores) * 1.08)
    else:
        ax.text(0.5, 0.5, "该方向无逐轮记录", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("冠军在门控集上的 Macro-F1")
    ax.set_title("闭环逐轮进展（%d 个决策周期，门控集 F1）" % len(ledger))
    ax.legend(loc="lower right", fontsize=10)

    # Panel 2: the final comparison on the metrics that matter.
    ax = axes[1]
    labels = ["静态基线", "智训引擎"]
    macro = [result["baseline"]["macro_f1"], result["system"]["macro_f1"]]
    accuracy = [result["baseline"]["accuracy"], result["system"]["accuracy"]]
    positions = np.arange(2)
    ax.bar(positions - 0.19, macro, 0.36, label="Macro-F1（类别均衡）", color=BLUE)
    ax.bar(positions + 0.19, accuracy, 0.36, label="Accuracy（整体）", color=LIGHT)
    for index in range(2):
        ax.text(index - 0.19, macro[index], "%.4f" % macro[index], ha="center",
                va="bottom", fontsize=11, fontweight="bold")
        ax.text(index + 0.19, accuracy[index], "%.4f" % accuracy[index], ha="center",
                va="bottom", fontsize=11)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("分数")
    ax.set_title("最终对比：Macro-F1 提升 %.2f%%" % result["primary_gain_pct"])
    ax.legend(loc="lower right", fontsize=10)

    figure.suptitle("A 方向气体分类（6 类）：闭环进展与最终效果", fontsize=15, y=1.04)
    figure.tight_layout()
    path = out / "03_分类方向_闭环进展与最终效果.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_ledger(plt, out: Path, reports: dict) -> Path:
    """The closed loop's ledger: what was tried, what it claimed, what happened.

    A table, not a bar chart: the point of the ledger is the *reasoning*
    (the hypothesis, its claim, its falsification condition and the
    verdict), and a table is the only form that carries all four at a
    readable size.
    """
    if not reports:
        return None

    profile = "B"
    if profile not in reports:
        profile = sorted(reports)[0]
    ledger = reports[profile].get("ledger") or []
    if not ledger:
        return None

    # The rows are assembled first, then drawn as a real table.  The
    # earlier version positioned every cell by hand, which meant the
    # columns drifted apart as the text got longer - a claim that ran
    # long would run into the gain column.  A table gives the columns
    # fixed widths and the text a box to sit in.
    verdicts = [str(entry.get("decision", entry.get("action", "?"))).lower()
                for entry in ledger]
    palette = {"promote": GREEN, "reject": ACCENT}
    default = GREY

    def _wrap(text, limit):
        """Break a long claim into two short lines."""
        text = str(text or "—")
        if len(text) <= limit:
            return text
        cut = text.rfind("。", 0, limit)
        if cut < limit // 2:
            cut = limit
        else:
            cut += 1
        return text[:cut] + "\n" + text[cut:cut + limit]

    header = ["周期", "假设", "主张", "收益", "裁决"]
    body = []
    for entry, verdict in zip(ledger, verdicts):
        body.append([
            str(entry.get("cycle", "")),
            str(entry.get("hypothesis", "—")),
            _wrap(entry.get("claim", "—"), 46),
            "%+.1f" % float(entry.get("gain", 0.0) or 0.0),
            "晋级" if verdict == "promote" else
            ("拒绝" if verdict == "reject" else verdict),
        ])

    figure, ax = plt.subplots(figsize=(15.0, 1.05 + 0.72 * len(body)))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, loc="upper center",
                      cellLoc="left", colLoc="left",
                      colWidths=[0.045, 0.175, 0.545, 0.075, 0.075])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.85)

    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#f0f3f6")
            cell.set_text_props(weight="bold", color="#1a3d5c")
            cell.get_text().set_ha("left")
            continue
        cell.set_facecolor("white")
        verdict = verdicts[row - 1]
        color = palette.get(verdict, default)
        if column == 0:
            cell.set_text_props(color="#666666")
        elif column == 1:
            cell.set_text_props(color="#1a1a1a")
        elif column == 2:
            cell.set_text_props(color="#666666", fontsize=8.8)
        elif column in (3, 4):
            cell.set_text_props(color=color, weight="bold")
        cell.get_text().set_ha("left")
        cell.get_text().set_va("center")

    # The legend: what the colours mean.
    ax.text(0.01, 0.005,
            "绿 = 晋级（假设成立，配方叠加到冠军）    "
            "红 = 拒绝（假设被证伪，冠军不变）    "
            "灰 = 不动作（证据不足，保守等待）",
            transform=ax.transAxes, fontsize=10, color="#444444", va="bottom")

    figure.suptitle("%s 方向闭环决策账本：每一轮都留下假设、主张、证伪条件与裁决"
                    % profile, fontsize=14.5, y=0.99)
    path = out / "04_闭环决策账本.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_online_curve(plt, out: Path, curve: dict) -> Path:
    """The evo package's online curve: the static baseline degrading."""
    if not curve:
        return None
    adaptive = curve.get("adaptive") or []
    static = curve.get("static_baseline") or []
    if not adaptive or not static:
        return None

    figure, ax = plt.subplots(figsize=(10.5, 5.2))
    windows = np.arange(1, len(adaptive) + 1)
    ax.plot(windows, adaptive, marker="o", linewidth=2.6, color=BLUE,
            label="自适应（本系统）", zorder=3)
    ax.plot(windows, static, marker="s", linewidth=2.2, color=GREY,
            label="静态基线（冻结）", zorder=2)
    ax.fill_between(windows, adaptive, static, where=np.asarray(adaptive) >= np.asarray(static),
                    color=GREEN, alpha=0.12, label="本系统领先区间")

    best = int(np.argmax(adaptive)) + 1
    ax.scatter([best], [adaptive[best - 1]], s=190, color=ACCENT, zorder=5,
               edgecolors="white", linewidths=1.8)
    ax.annotate("最佳：%.1f\n（第 %d 个窗口）" % (adaptive[best - 1], best),
                xy=(best, adaptive[best - 1]), xytext=(best + 0.25, adaptive[best - 1] - 90),
                fontsize=11, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=1.6))

    ax.set_xlabel("流式窗口（时间顺序）")
    ax.set_ylabel("分数（−RMSE，越小越好）")
    ax.set_title("在线自适应曲线：静态基线一路退化，自适应系统持续跟踪并刷新最优")
    ax.set_xticks(windows)
    ax.legend(loc="lower right", fontsize=11)
    ax.margins(y=0.16)

    path = out / "05_在线自适应曲线.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_per_unit(plt, out: Path, verification: dict, reports: dict) -> Path:
    """The per-unit error on the regression tracks, side by side."""
    results = [r for r in (verification or {}).get("results", [])
               if r["metric"] == "rmse"]
    if not results:
        return None

    figure, axes = plt.subplots(1, len(results), figsize=(6.4 * len(results), 4.8),
                                 squeeze=False)
    for index, result in enumerate(results):
        ax = axes[0][index]
        units = list(result["per_unit_system"].keys())
        base_unit = [result["per_unit_baseline"].get(u, {}).get("rmse", 0.0)
                     for u in units]
        system_unit = [result["per_unit_system"][u]["rmse"] for u in units]

        # The units are grouped before anything is drawn.  Profile C's
        # validation set holds 111 engines; one bar and one label each
        # turns the axis into a smear and hides the very thing the chart
        # is about.  Grouping by the sub-dataset (FD001 ... FD004) gives
        # four readable columns, and the per-engine spread is shown as
        # the error bar rather than as 111 separate marks.
        groups = {}
        for unit, base, system in zip(units, base_unit, system_unit):
            group = unit.split("#", 1)[0] if "#" in unit else (
                unit.split("_")[0] if "_" in unit else unit)
            groups.setdefault(group, []).append((base, system))

        labels = list(groups.keys())
        base_means = [float(np.mean([b for b, _ in members]))
                      for members in groups.values()]
        system_means = [float(np.mean([s for _, s in members]))
                        for members in groups.values()]
        # The spread inside the group: the standard deviation of the
        # per-unit RMSE, which is where the hard units live.
        system_spread = [float(np.std([s for _, s in members]))
                         for members in groups.values()]

        positions = np.arange(len(labels))
        width = 0.34
        bars_base = ax.bar(positions - width / 2, base_means, width,
                           label="静态基线", color=GREY)
        bars_system = ax.bar(positions + width / 2, system_means, width,
                              label="智训引擎", color=BLUE,
                              yerr=system_spread, capsize=4,
                              error_kw=dict(ecolor="#1b4a6b", lw=1.4))

        for position, (base, system, spread) in enumerate(
                zip(base_means, system_means, system_spread)):
            gain = (base - system) / base * 100 if base else 0.0
            ax.text(position, max(base, system) + spread * 0.35, "%+.1f%%" % gain,
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                    color=GREEN if gain > 0 else ACCENT)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("RMSE（组内均值）")
        ax.set_title("%s 方向：按%s分组的误差"
                     % (result["profile"],
                        "子数据集" if "#" in units[0] else "轴承"))
        ax.legend(loc="upper left", fontsize=9.5)
        ax.margins(y=0.24)

        # How many units each column stands for - the reader should not
        # have to guess whether FD004 is one engine or thirty.
        note = "每组 %s 个单元；误差棒 = 组内标准差" % (
            "/".join(str(len(members)) for members in groups.values()))
        ax.text(0.5, -0.20, note, transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#666666")

    figure.suptitle("逐单元误差分解：剩余误差集中在少数长寿命单元上",
                     fontsize=15, y=1.04)
    figure.tight_layout()
    path = out / "06_逐单元误差分解.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figureuse")
    args = parser.parse_args(argv)

    plt = _style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verification = _read(ROOT / "artifacts" / "rmse_verification.json")
    curve = _read(ROOT / "artifacts" / "official" / "online_curve.json")
    reports = {}
    for profile in ("A", "B", "C"):
        report = _read(ROOT / "runs" / profile / "run_report.json")
        if report:
            reports[profile] = report

    written = []
    for builder in (lambda: figure_headline(plt, out, verification, []),
                    lambda: figure_rmse(plt, out, verification),
                    lambda: figure_classification(plt, out, verification, reports),
                    lambda: figure_ledger(plt, out, reports),
                    lambda: figure_online_curve(plt, out, curve),
                    lambda: figure_per_unit(plt, out, verification, reports)):
        try:
            path = builder()
        except Exception as exc:
            print("  (a figure failed: %s)" % exc)
            continue
        if path:
            written.append(path)
            print("  wrote %s" % path)

    print("%d figures written to %s" % (len(written), out))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
