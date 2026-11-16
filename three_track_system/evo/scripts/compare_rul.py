"""赛题 B (轴承 RUL) 对照实验 —— 定量证明自进化闭环的价值。

对比四种系统 (同数据、同轴承留出、同评估口径):
  E0  冻结静态基线  : 只训练一次, 永不更新 (门槛③对照)
  E1  定时重训      : 每轮都重训, 无漂移检测 (烧预算)
  E2  漂移触发重训  : 有监测+归因, 无策略层探索 (固定超参)
  E3  完整系统      : 模型层 + 策略层 (本方案)

评估口径 (与赛题一致):
  * 流上逐窗口 RMSE (分钟), 越低越好
  * tail = 流后 1/3 段的平均 RMSE —— 最能反映长期稳定性
  * 效率 = RMSE 改善 / 加权成本
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evo.config import load_config
from evo.harness import Harness

CONFIG = str(ROOT / "configs/rul_bearing.yaml")


def run_variant(name: str, mutate=None, seeds=(20261110, 20261111, 20261112)) -> dict:
    results = []
    for seed in seeds:
        cfg = load_config(CONFIG)
        cfg.set("runtime.seed", seed)
        cfg.set("paths.artifacts_root", f"artifacts/rulcmp_{name}_s{seed}")
        cfg.set("paths.audit_log", f"artifacts/rulcmp_{name}_s{seed}/audit/audit.jsonl")
        cfg.set("paths.shadow_root", f"artifacts/rulcmp_{name}_s{seed}/shadow")
        cfg.set("experiment.local_dir", f"artifacts/rulcmp_{name}_s{seed}/exp")
        if mutate:
            mutate(cfg)

        h = Harness(cfg, use_synthetic=False)
        s = h.run()
        results.append(s.to_dict())
        print(
            f"  [{name} seed={seed}] tail_RMSE={-s.tail_score:8.1f} "
            f"base_tail_RMSE={-s.baseline_tail_score:8.1f} "
            f"iter={s.iterations} approved={s.updates_approved} eff={s.efficiency:.4f}"
        )
    agg = {}
    for key in (
        "baseline_score",
        "final_score",
        "tail_score",
        "baseline_tail_score",
        "total_gpu_seconds",
        "total_labels",
        "efficiency",
        "updates_approved",
        "iterations",
    ):
        vals = [r[key] for r in results]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def make_static(cfg):
    """E0: 冻结静态基线 —— 部署后永不更新。"""
    cfg.set("policy.updates_enabled", False)


def make_periodic(cfg):
    """E1: 定时重训 (无漂移门控)。"""
    cfg.set("policy.periodic_retrain", True)
    cfg.set("safety.max_candidates_per_iter", 1)


def make_no_bandit(cfg):
    """E2: 漂移触发重训 + 固定超参 (无策略层探索)。"""
    cfg.set("policy.exploration_rate", 0.0)
    cfg.set("safety.max_candidates_per_iter", 1)
    cfg.set("policy.param_space", {
        "psi_trigger": [0.1],
        "min_gain": [25.0],
        "retrain_window": [2000],
        "lr": [0.05],
        "num_leaves": [31],
    })


def make_full(cfg):
    """E3: 完整系统。"""
    pass


VARIANTS = [
    ("E0_static", make_static),
    ("E1_periodic", make_periodic),
    ("E2_no_bandit", make_no_bandit),
    ("E3_full", make_full),
]


def main() -> int:
    table = {}
    for name, fn in VARIANTS:
        print(f"\n=== {name} ===")
        table[name] = run_variant(name, fn)

    print("\n" + "=" * 104)
    print(
        f"{'system':<16}{'tail RMSE':>12}{'base tail':>12}{'improve':>10}"
        f"{'iter':>7}{'approved':>10}{'labels':>9}{'gpu(s)':>9}{'eff':>12}"
    )
    print("-" * 104)
    for name, agg in table.items():
        tail = -agg["tail_score"]["mean"]
        base = -agg["baseline_tail_score"]["mean"]
        imp = (base - tail) / base * 100 if base else 0.0
        print(
            f"{name:<16}{tail:>12.1f}{base:>12.1f}{imp:>9.1f}%"
            f"{agg['iterations']['mean']:>7.1f}"
            f"{agg['updates_approved']['mean']:>10.1f}"
            f"{agg['total_labels']['mean']:>9.0f}"
            f"{agg['total_gpu_seconds']['mean']:>9.1f}"
            f"{agg['efficiency']['mean']:>12.4f}"
        )
    print("=" * 104)
    print("(tail RMSE 越低越好; base tail = 冻结静态基线的尾段 RMSE; improve = 相对基线的改善%)")

    out = ROOT / "artifacts/rul_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
