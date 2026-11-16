"""官方静态基线对照 —— 与赛题包 checkpoint 直接比对。

**为什么必须做这个对照**:
    赛题门槛原文是「最终模型性能不得低于官方提供的静态基线」。
    我们自己训练的静态基线只是**参照**, 真正的门槛线是官方
    `checkpoint.npz` (NumPy MLP, 865 参数) 在同一评估集上的成绩。

    只有直接跑官方 checkpoint, 才能回答评委最可能问的问题:
    "你的系统比官方基线好多少?"

用法:
    python scripts/compare_official_baseline.py \
        --data-root "D:/GY/deepseek/cpu-b-rul-20261110-v1/public" \
        --package   "D:/GY/deepseek/cpu-b-rul-20261110-v1" \
        --artifacts artifacts/official4

输出:
    * 官方 checkpoint 在验证集上的 RMSE / MAE
    * 我们的静态基线与自适应最终模型的 RMSE / MAE
    * 相对官方基线的提升百分比
    * 结果落盘到 artifacts/official_baseline_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evo.config import load_config
from evo.harness import Harness


def _official_metrics(package: str, data_root: str) -> dict:
    """在官方验证集上跑官方 checkpoint。

    优先直接复用赛题包的 `baselines` 模块 (保证与官方完全一致的
    flatten / 预测 / 评分逻辑), 而不是自己复写一遍 ——
    自己复写很容易在细节上偏离官方口径, 导致对比不公平。
    """
    package_path = Path(package)
    baselines_dir = package_path / "baselines"
    if not baselines_dir.exists():
        return {"available": False, "reason": f"未找到 {baselines_dir}"}

    sys.path.insert(0, str(package_path))
    try:
        from baselines.data import read_dataset
        from baselines.model import Model
    except Exception as exc:  # pragma: no cover - 环境问题
        return {"available": False, "reason": f"导入官方 baselines 失败: {exc}"}

    ckpt = package_path / "checkpoint.npz"
    if not ckpt.exists():
        return {"available": False, "reason": f"未找到官方检查点 {ckpt}"}

    valid_file = Path(data_root) / "validation.jsonl"
    if not valid_file.exists():
        return {"available": False, "reason": f"未找到验证集 {valid_file}"}

    try:
        model = Model.load(ckpt)
        data = read_dataset(str(valid_file), "regression")
    except Exception as exc:
        return {"available": False, "reason": f"官方基线执行失败: {exc}"}

    pred = np.asarray(model.predict(data["x"]), dtype=float)
    truth = np.asarray(data["y"], dtype=float)
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mae = float(np.mean(np.abs(pred - truth)))
    return {
        "available": True,
        "rmse": rmse,
        "mae": mae,
        "n_samples": int(len(truth)),
        "checkpoint": str(ckpt),
        "parameter_count": int(
            (model.meta or {}).get("parameter_count", 0) or 0
        ),
    }


def _our_metrics(artifacts: str) -> dict:
    """从我们的运行产物里读出静态基线与自适应最终模型的成绩。"""
    root = Path(artifacts)
    summary_path = root / "run_summary.json"
    if not summary_path.exists():
        return {"available": False, "reason": f"未找到 {summary_path}"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = summary.get("gates") or {}
    holdout = gates.get("holdout_compare") or {}
    if not holdout:
        return {
            "available": False,
            "reason": "运行产物里没有 holdout_compare (需用官方数据跑一次 run)",
        }
    return {
        "available": True,
        "adaptive": {
            "rmse": -float(holdout["adaptive"]),
            "mae": float(holdout["adaptive_metrics"].get("mae", 0.0)),
        },
        "static_baseline": {
            "rmse": -float(holdout["baseline"]),
            "mae": float(holdout["baseline_metrics"].get("mae", 0.0)),
        },
        "n_samples": int(holdout.get("n_samples", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--package",
        default=r"D:\GY\deepseek\cpu-b-rul-20261110-v1",
        help="赛题包根目录 (含 checkpoint.npz 与 baselines/)",
    )
    ap.add_argument(
        "--data-root",
        default=r"D:\GY\deepseek\cpu-b-rul-20261110-v1\public",
        help="公开数据目录 (含 validation.jsonl)",
    )
    ap.add_argument(
        "--artifacts",
        default="artifacts/official4",
        help="我们的运行产物目录 (含 run_summary.json)",
    )
    ap.add_argument("--config", default="configs/rul_bearing.yaml")
    args = ap.parse_args()

    official = _official_metrics(args.package, args.data_root)
    ours = _our_metrics(args.artifacts)

    report: dict = {
        "official": official,
        "ours": ours,
    }

    print("=" * 76)
    print("赛题 B 静态基线对照 (评估集: 官方验证集, 整根轴承隔离)")
    print("=" * 76)

    if official.get("available"):
        print(
            f"官方静态基线 (NumPy MLP, {official.get('parameter_count')} 参数): "
            f"RMSE={official['rmse']:.2f} 分钟  MAE={official['mae']:.2f} 分钟 "
            f"(n={official['n_samples']})"
        )
    else:
        print(f"官方静态基线不可用: {official.get('reason')}")

    if ours.get("available"):
        sb = ours["static_baseline"]
        ad = ours["adaptive"]
        print(
            f"我们的静态基线 (E0):                    "
            f"RMSE={sb['rmse']:.2f} 分钟  MAE={sb['mae']:.2f} 分钟"
        )
        print(
            f"我们的自适应最终模型 (E3):              "
            f"RMSE={ad['rmse']:.2f} 分钟  MAE={ad['mae']:.2f} 分钟"
        )
        if official.get("available"):
            rmse_gain = (official["rmse"] - ad["rmse"]) / official["rmse"] * 100
            mae_gain = (official["mae"] - ad["mae"]) / official["mae"] * 100
            print("-" * 76)
            print(
                f"自适应系统相对**官方静态基线**: RMSE 提升 {rmse_gain:+.1f}%  "
                f"MAE 提升 {mae_gain:+.1f}%"
            )
            report["gain_vs_official"] = {
                "rmse_pct": round(rmse_gain, 2),
                "mae_pct": round(mae_gain, 2),
                "passed": bool(ad["rmse"] <= official["rmse"]),
            }
    else:
        print(f"我们的成绩不可用: {ours.get('reason')}")

    out = Path(args.artifacts) / "official_baseline_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
