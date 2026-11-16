"""全场景验证 — 覆盖赛题的三条基础门槛 + 鲁棒性场景。

场景:
  S1 基准闭环          : 无人工干预跑完全流程
  S2 标签延迟          : label_delay=500, 标签不可提前看到
  S3 标注配额紧张      : max_label_quota=300, 应转为复用历史标注
  S4 LLM 不可用        : 无 PyTRiO 服务, 应规则降级且跑完
  S5 极端漂移          : drift_strength=0.6, 验证回滚/熔断有效
  S6 静态基线对照      : 验证门槛③ (不低于静态基线)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evo.audit import AuditLog
from evo.config import load_config
from evo.experiment import ExperimentRecorder
from evo.harness import Harness


def run_case(name: str, mutate=None) -> dict:
    cfg = load_config()
    cfg.set("runtime.seed", 11)
    cfg.set("runtime.window_size", 500)
    cfg.set("safety.max_iterations", 25)
    cfg.set("budget.max_label_quota", 2000)
    cfg.set("paths.artifacts_root", f"artifacts/scen_{name}")
    cfg.set("paths.audit_log", f"artifacts/scen_{name}/audit/audit.jsonl")
    cfg.set("paths.shadow_root", f"artifacts/scen_{name}/shadow")
    cfg.set("experiment.local_dir", f"artifacts/scen_{name}/exp")
    if mutate:
        mutate(cfg)

    h = Harness(cfg, use_synthetic=True)
    s = h.run()

    audit_ok = AuditLog(cfg).verify()["valid"]
    comp = ExperimentRecorder(cfg).compliance_report()
    return {
        "case": name,
        "iterations": s.iterations,
        "drifts": s.drifts_detected,
        "approved": s.updates_approved,
        "halted": s.halted,
        "tail_score": round(s.tail_score, 4),
        "base_tail": round(s.baseline_tail_score, 4),
        "beat_baseline": s.tail_score >= s.baseline_tail_score,
        "gpu_s": round(s.total_gpu_seconds, 2),
        "labels": s.total_labels,
        "audit_valid": audit_ok,
        "records_ok": comp["compliant"],
        "summary_exists": (cfg.path("paths.artifacts_root") / "run_summary.json").exists(),
    }


def m_delay(cfg):
    cfg.set("runtime.label_delay", 500)


def m_tight_label(cfg):
    cfg.set("budget.max_label_quota", 300)


def m_no_llm(cfg):
    import os

    for k in ("PYTRIO_BASE_URL", "PYTRIO_API_KEY", "PYTRIO_MODEL"):
        os.environ.pop(k, None)


def m_hard_drift(cfg):
    cfg.set("data.synthetic.drift_strength", 0.6)
    cfg.set("data.synthetic.noise_growth", 2.5)


def m_static(cfg):
    cfg.set("policy.updates_enabled", False)


CASES = [
    ("S1_baseline", None),
    ("S2_label_delay", m_delay),
    ("S3_tight_label", m_tight_label),
    ("S4_no_llm", m_no_llm),
    ("S5_hard_drift", m_hard_drift),
    ("S6_static_ref", m_static),
]


def main() -> int:
    results = []
    for name, fn in CASES:
        r = run_case(name, fn)
        results.append(r)
        print(
            f"{name:<16} iters={r['iterations']:>3} drift={r['drifts']:>3} "
            f"approved={r['approved']:>3} tail={r['tail_score']:.4f} "
            f"(base {r['base_tail']:.4f}) pass={'OK' if r['beat_baseline'] else 'NG'} "
            f"audit={'OK' if r['audit_valid'] else 'NG'} "
            f"records={'OK' if r['records_ok'] else 'NG'}"
        )

    print("\n" + "=" * 92)
    all_ok = all(
        r["beat_baseline"] and r["audit_valid"] and r["records_ok"] and r["summary_exists"]
        for r in results
    )
    print(f"all gates passed: {'YES' if all_ok else 'NO'}")
    out = Path("artifacts/scenarios.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
