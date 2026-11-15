"""Render all six-class results from the immutable first run; no fitting."""
import json
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    result = json.loads((folder / "run_01/results.json").read_text())
    lines = ["# 第九轮完整分类指标", "", "数据来源：run_01/results.json。所有对照均重新在同一训练侧拟合；非原始全量训练检查点成绩。",
             "", "每类单元格依次为 precision / recall / F1；无预测时 precision 按参考评分器记 0，是否有定义见原始 JSON。每个评估域中各候选的行完全相同。", ""]
    for fold in result["updates"]:
        rows = dict(fold["baselines"])
        rows.update({f"C_{int(k)*6}labels": v["metrics"] for k, v in fold["budgets"].items()})
        rows.update({f"C_{k}_12labels": v["metrics"] for k, v in fold["missing"].items()})
        for name, stages in fold["orders"].items():
            for i, stage in enumerate(stages, 1):
                rows[f"{name}_stage{i}"] = stage["metrics"]
        rows["pooled_24labels"] = fold["pooled"]["metrics"]
        for domain in ("old_source", "clean", "composite"):
            support_key = "old_test" if domain == "old_source" else "evaluation"
            lines.extend([f"## Fold {fold['fold']} / {domain}", "",
                          f"支持数（Et / Ethylene / Ammonia / Ac / Acetone / Toluene）：{fold['partition']['supports'][support_key]}。", "",
                          "| 条件 | Macro-F1 | Accuracy | 乙醇 P/R/F1 | 乙烯 P/R/F1 | 氨 P/R/F1 | 乙醛 P/R/F1 | 丙酮 P/R/F1 | 甲苯 P/R/F1 |",
                          "|---|---:|---:|---|---|---|---|---|---|"])
            for name, domains in rows.items():
                metrics = domains[domain]
                cells = [name, f"{metrics['macro_f1']:.6f}", f"{metrics['accuracy']:.6f}"]
                for label in ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"):
                    cells.append(" / ".join(f"{metrics['per_class'][label][key]:.4f}" for key in ("precision", "recall", "f1")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    (folder / "RESULT_TABLES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
