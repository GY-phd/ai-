"""RUL 场景参数扫描 —— 找出影子集大小 / 探索率的最优组合。

这是"实验规范与分析深度"的直接体现: 关键超参不应拍脑袋, 而应扫出来。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evo.config import load_config
from evo.harness import Harness

CONFIG = str(ROOT / "configs/rul_bearing.yaml")


def trial(tag: str, **over) -> dict:
    cfg = load_config(CONFIG)
    cfg.set("paths.artifacts_root", f"artifacts/sweep/{tag}")
    cfg.set("paths.audit_log", f"artifacts/sweep/{tag}/audit/audit.jsonl")
    cfg.set("paths.shadow_root", f"artifacts/sweep/{tag}/shadow")
    cfg.set("experiment.local_dir", f"artifacts/sweep/{tag}/exp")
    for k, v in over.items():
        cfg.set(k, v)
    h = Harness(cfg, use_synthetic=False)
    s = h.run()
    return {
        "tag": tag,
        "tail_rmse": -s.tail_score,
        "base_tail_rmse": -s.baseline_tail_score,
        "gpu": s.total_gpu_seconds,
        "labels": s.total_labels,
        "approved": s.updates_approved,
        "iters": s.iterations,
        "eff": s.efficiency,
    }


def main() -> int:
    print(f"{'tag':<34}{'tail RMSE':>11}{'gpu(s)':>9}{'labels':>8}{'eff':>10}")
    print("-" * 72)
    rows = []

    # -- 扫描 1: 影子集大小 (探索率固定 0.25) --
    for holdout in (100, 200, 300, 500):
        r = trial(
            f"holdout{holdout}",
            **{
                "safety.shadow_holdout": holdout,
                "safety.shadow_min_steps": int(holdout * 0.6),
            },
        )
        rows.append(r)
        print(
            f"{'holdout=' + str(holdout):<34}{r['tail_rmse']:>11.1f}"
            f"{r['gpu']:>9.1f}{r['labels']:>8}{r['eff']:>10.2f}"
        )

    # -- 扫描 2: 探索率 (影子集取最优) --
    best_row = min(rows, key=lambda r: r["tail_rmse"])
    holdout = int(best_row["tag"].replace("holdout", ""))
    print(f"\n(best holdout = {holdout}, now sweeping exploration_rate)")
    print("-" * 72)
    for rate in (0.0, 0.1, 0.25, 0.5):
        r = trial(
            f"rate{rate}_h{holdout}",
            **{
                "policy.exploration_rate": rate,
                "safety.shadow_holdout": holdout,
                "safety.shadow_min_steps": int(holdout * 0.6),
            },
        )
        rows.append(r)
        print(
            f"{'rate=' + str(rate) + ' h=' + str(holdout):<34}{r['tail_rmse']:>11.1f}"
            f"{r['gpu']:>9.1f}{r['labels']:>8}{r['eff']:>10.2f}"
        )

    # -- 扫描 3: 训练窗口 --
    print("\n(sweeping retrain_window)")
    print("-" * 72)
    for win in (500, 1000, 2000, 4000):
        r = trial(
            f"win{win}",
            **{
                "policy.param_space.retrain_window": [win],
                "policy.exploration_rate": 0.15,
                "safety.shadow_holdout": holdout,
                "safety.shadow_min_steps": int(holdout * 0.6),
            },
        )
        rows.append(r)
        print(
            f"{'window=' + str(win):<34}{r['tail_rmse']:>11.1f}"
            f"{r['gpu']:>9.1f}{r['labels']:>8}{r['eff']:>10.2f}"
        )

    best = min(rows, key=lambda r: r["tail_rmse"])
    print(f"\nbest: {best['tag']}  tail_rmse={best['tail_rmse']:.1f}  eff={best['eff']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
