"""对照实验 — 定量证明"完整系统 > 静态基线"。

对比四种系统 (同种子、同数据流、同评估口径):
  E0  冻结静态基线      : 只训练一次, 永不更新 (赛题门槛③的对照)
  E1  定时重训          : 每轮都重训, 无漂移检测 (烧预算)
  E2  漂移触发重训      : 有监测+归因, 无策略层 bandit
  E3  完整系统          : 模型层 + 策略层 (本方案)

输出对照表 + 成本收益 (efficiency), 可直接放进答辩材料。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evo.config import load_config
from evo.harness import Harness
from evo.models import compute_metrics


def run_variant(name: str, mutate, seeds=(11, 22, 33), **cfg_over):
    """跑一个变体, 多随机种子取均值。"""
    results = []
    for seed in seeds:
        cfg = load_config()
        cfg.set("runtime.seed", seed)
        cfg.set("runtime.offline", True)
        for k, v in cfg_over.items():
            cfg.set(k, v)
        if mutate:
            mutate(cfg)
        cfg.set("paths.artifacts_root", f"artifacts/exp_{name}_s{seed}")
        cfg.set("paths.audit_log", f"artifacts/exp_{name}_s{seed}/audit/audit.jsonl")
        cfg.set("paths.shadow_root", f"artifacts/exp_{name}_s{seed}/shadow")
        cfg.set("experiment.local_dir", f"artifacts/exp_{name}_s{seed}/exp")

        h = Harness(cfg, use_synthetic=True)
        s = h.run()
        results.append(s.to_dict())
        print(f"  [{name} seed={seed}] tail={s.tail_score:.4f} "
              f"base_tail={s.baseline_tail_score:.4f} eff={s.efficiency:.6f}")
    agg = {}
    for key in ("baseline_score", "final_score", "tail_score", "baseline_tail_score",
                "total_gpu_seconds", "total_labels", "efficiency", "updates_approved"):
        vals = [r[key] for r in results]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    # 种子级明细 —— 便于答辩时逐种子核对, 也用于配对统计检验
    agg["per_seed"] = {
        str(seed): {
            "tail_score": r["tail_score"],
            "baseline_tail_score": r["baseline_tail_score"],
            "efficiency": r["efficiency"],
            "updates_approved": r["updates_approved"],
            "total_gpu_seconds": r["total_gpu_seconds"],
            "total_labels": r["total_labels"],
        }
        for seed, r in zip(seeds, results)
    }
    return agg


def paired_test(a_vals, b_vals) -> dict:
    """配对检验 (同种子下两方案的差值)。

    种子数少 (3 个), 配对 t 检验的检验力很低, 因此同时报告
    Wilcoxon 符号秩检验与逐种子胜率 —— 三者共同给出结论, 避免
    只凭单一 p 值下判断。
    """
    from scipy import stats

    d = np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)
    wins = int(np.sum(d > 0))
    res = {
        "mean_diff": float(np.mean(d)),
        "std_diff": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        "wins": wins,
        "n": int(len(d)),
    }
    if len(d) > 1:
        res["t_pvalue"] = float(stats.ttest_rel(a_vals, b_vals).pvalue)
        try:
            res["wilcoxon_pvalue"] = float(
                stats.wilcoxon(d, alternative="greater").pvalue
            )
        except Exception:
            res["wilcoxon_pvalue"] = None
    else:
        res["t_pvalue"] = None
        res["wilcoxon_pvalue"] = None
    return res


# ---------------------------------------------------------------- 变体定义

def make_static(cfg):
    """E0: 冻结静态基线 —— 部署后永不更新。

    显式关闭主动更新, 但**保留完整迭代轮次**,
    这样 E0 与其他变体在完全相同的窗口序列上被评估 (公平对照)。
    """
    cfg.set("policy.updates_enabled", False)


def make_periodic(cfg):
    """E1: 定时重训 (无漂移检测, 固定周期无条件更新)。

    注意两点:
      1. 用 `policy.periodic_retrain` 开关而不是"把 psi_trigger 设成极小值" ——
         后者仍受熔断/预算约束, 且策略层一旦调整 psi_trigger 就失效,
         变体会退化成 E2/E3, 导致对照失去意义。
      2. 固定模型超参 (与 E2 同), 唯一差别是**触发机制**,
         这样 E1 vs E2 的差值才能干净地归因到"漂移检测"这一模块。
    """
    cfg.set("policy.periodic_retrain", True)
    cfg.set("policy.exploration_rate", 0.0)
    cfg.set("safety.max_candidates_per_iter", 1)
    cfg.set("policy.param_space", {
        "psi_trigger": [0.1], "min_gain": [0.005],
        "retrain_window": [2000], "lr": [0.05], "num_leaves": [31],
    })


def make_no_bandit(cfg):
    """E2: 固定策略参数, 关闭策略层自改进 (单候选)。"""
    cfg.set("policy.max_param_shift", 0)
    cfg.set("policy.exploration_rate", 0.0)
    cfg.set("safety.max_candidates_per_iter", 1)
    cfg.set("policy.param_space", {
        "psi_trigger": [0.1], "min_gain": [0.005],
        "retrain_window": [2000], "lr": [0.05], "num_leaves": [31],
    })


def make_full(cfg):
    """E3: 完整系统, 不做修改。"""
    pass


VARIANTS = [
    ("E0_static", make_static),
    ("E1_periodic", make_periodic),
    ("E2_no_bandit", make_no_bandit),
    ("E3_full", make_full),
]


SEEDS = (11, 22, 33, 44, 55)


def main() -> int:
    over = {
        "runtime.window_size": 500,
        "safety.max_iterations": 30,
        "budget.max_label_quota": 2000,
    }
    table = {}
    for name, fn in VARIANTS:
        print(f"\n=== {name} ===")
        table[name] = run_variant(name, fn, seeds=SEEDS, **over)

    print("\n" + "=" * 96)
    print(f"{'系统':<14}{'尾段得分':>12}{'静态基线尾段':>14}{'相对提升':>12}"
          f"{'更新次数':>10}{'标注':>8}{'效率(eff)':>14}")
    print("-" * 96)
    for name, agg in table.items():
        t = agg["tail_score"]["mean"]
        bt = agg["baseline_tail_score"]["mean"]
        lift = (t - bt) / bt * 100 if bt else 0.0
        print(
            f"{name:<14}{t:>12.4f}{bt:>14.4f}{lift:>11.1f}%"
            f"{agg['updates_approved']['mean']:>10.1f}"
            f"{agg['total_labels']['mean']:>8.0f}"
            f"{agg['efficiency']['mean']:>14.6f}"
        )
    print("=" * 96)

    # ---- 配对显著性检验 (同种子, 逐种子配对) ----
    def per_seed_vec(name: str, metric: str):
        return [table[name]["per_seed"][str(s)][metric] for s in SEEDS]

    tests = {}
    for metric in ("tail_score", "efficiency"):
        for a, b in (("E3_full", "E0_static"),
                     ("E3_full", "E1_periodic"),
                     ("E3_full", "E2_no_bandit"),
                     ("E2_no_bandit", "E0_static")):
            key = f"{metric}:{a}_vs_{b}"
            tests[key] = paired_test(
                per_seed_vec(a, metric), per_seed_vec(b, metric)
            )

    print("\n配对显著性检验 (同种子配对, %d 个种子)" % len(SEEDS))
    print(f"{'对比':<44}{'平均差':>10}{'胜/总':>8}{'t检验p':>10}{'Wilcoxonp':>11}")
    print("-" * 96)
    for key, r in tests.items():
        t_p = f"{r['t_pvalue']:.4f}" if r["t_pvalue"] is not None else "n/a"
        w_p = f"{r['wilcoxon_pvalue']:.4f}" if r["wilcoxon_pvalue"] is not None else "n/a"
        wr = f"{r['wins']}/{r['n']}"
        print(f"{key:<44}{r['mean_diff']:>10.4f}{wr:>8}{t_p:>10}{w_p:>11}")
    print("-" * 96)
    print("注: 配对 t 检验在少量种子下检验力有限; 以逐种子胜率 + Wilcoxon 一并判断。")

    out = Path("artifacts/comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": table, "significance": tests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
