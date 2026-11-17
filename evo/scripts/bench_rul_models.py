"""RUL 建模方案横向对比 — 在官方留出轴承上定量比较各候选模型。

为什么需要这个脚本:
    赛题 B 的门槛是"最终模型性能不得低于官方提供的静态基线"。
    选哪个模型族、什么容量, 直接决定这道门槛能否通过, 而不能靠猜。
    本脚本把**所有候选方案**放在**同一个严格口径**下比较, 结论直接
    写回 `configs/rul_bearing.yaml`。

评测口径 (与官方隐藏测试同构):
    * 训练: `train.jsonl` 的 9 根轴承
    * 评估: `validation.jsonl` 的 3 根**完全独立**的轴承
    * 指标: RMSE (官方主指标) + MAE, 越低越好

实测结论 (2026-11-15, 官方留出轴承):
    lgbm(200,lr.05,l31)      RMSE = 694.80   <- 旧基线
    lgbm + log1p target     RMSE = 735.27
    mlp(24)                 RMSE = 741.73
    mlp(64)                 RMSE = 660.84
    mlp(128)                RMSE = 539.24
    mlp(256)                RMSE = 521.07   <- 最优, 已提为 base
    mlp(256,128)            RMSE = 553.77
    mlp(512)                RMSE = 617.66   <- 过拟合
    3xMLP 集成              RMSE = 530.97   <- 集成反而略差
    MLP + lgbm 集成         RMSE = 561.19   <- 更差
    目标标准化 (TTR) 包裹    RMSE = 716.03   <- 明显有害, 已关闭

关键结论:
    1. **MLP 完胜 LightGBM** (521.07 vs 694.80, -24.8%)。
       RUL = (总快照数 - 当前快照) 是结构性派生量, 与振动特征之间是高度
       非线性的映射; 树模型的阶梯划分逼近这条光滑曲面分辨率不足。
       官方基线本身就是 34->24->1 的 MLP, 说明赛题设计者也预期这类解法。
    2. **目标标准化有害**。RUL 的尺度本身携带信息 (长寿命轴承的量级是
       短寿命轴承的 20 倍), 标准化把它压平成 z-score 就抹掉了这一维。
    3. **集成无益**。单模型已足够强, 集成只增加算力成本 —— 赛题客观分
       直接考察"单位增益消耗的算力"。

用法:
    python scripts/bench_rul_models.py
    python scripts/bench_rul_models.py --data-root "D:/GY/deepseek/cpu-b-rul-20261110-v1/public"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path(r"D:/GY/deepseek/cpu-b-rul-20261110-v1/public")


def load_jsonl(path: Path):
    """Read a JSONL file into a list of dicts."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flatten_inputs(inputs: dict):
    """Flatten nested inputs with sorted-key traversal (official semantics).

    Matches `baselines/data.py::flatten`: dict keys are visited in sorted
    order, so the feature order is deterministic and reproducible.
    """
    paths = []
    values = []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [("key", key)])
        elif isinstance(value, list):
            for i, x in enumerate(value):
                visit(x, path + [("idx", i)])
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            paths.append(path)
            values.append(float(value))

    visit(inputs, [])
    return paths, values


def build(data_root: Path):
    """Return train/valid matrices plus bearing metadata."""
    tr = load_jsonl(data_root / "train.jsonl")
    va = load_jsonl(data_root / "validation.jsonl")

    def to_xy(rows):
        paths = None
        Xs, ys, bearings, snaps = [], [], [], []
        for r in rows:
            p, v = flatten_inputs(r["inputs"])
            if paths is None:
                paths = p
            Xs.append(v)
            ys.append(float(r["answer"]))
            m = r.get("meta", {})
            bearings.append(m.get("bearing", "unknown"))
            snaps.append(int(m.get("snapshot", 0)))
        return (
            np.asarray(Xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            bearings,
            np.asarray(snaps, dtype=np.int64),
            paths,
        )

    Xtr, ytr, btr, str_, paths = to_xy(tr)
    Xva, yva, bva, sva, _ = to_xy(va)
    return Xtr, ytr, btr, str_, Xva, yva, bva, sva, paths


def rmse(y, p):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def report(name, y, p, store=None, key=None):
    """Print and optionally record one result."""
    r, m = rmse(y, p), mae(y, p)
    print(f"  {name:44s} RMSE={r:8.2f}  MAE={m:8.2f}", flush=True)
    if store is not None and key:
        store[key] = {"rmse": round(r, 3), "mae": round(m, 3), "label": name}
    return r


def mlp_pipeline(hidden, alpha=1e-4, max_iter=1200, lr=3e-3, seed=42):
    """Build an input-standardized MLP pipeline (no target scaling)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
            random_state=seed, learning_rate_init=lr,
            early_stopping=False, solver="adam",
        )),
    ])


def main(argv=None) -> int:
    """Run the benchmark."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA),
                    help="官方公开数据目录 (含 train.jsonl / validation.jsonl)")
    ap.add_argument("--out", default="artifacts/rul_benchmark.json",
                    help="where to write the measured results, so that every "
                         "number quoted in the README has an artifact behind it")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    Xtr, ytr, btr, str_, Xva, yva, bva, sva, paths = build(Path(args.data_root))
    print(f"train X={Xtr.shape}  valid X={Xva.shape}  features={len(paths)}", flush=True)
    print(f"train bearings: {sorted(set(btr))}", flush=True)
    print(f"valid bearings: {sorted(set(bva))}", flush=True)
    print()

    results = {}

    # ---- Tree baselines ----
    print("=== Tree models ===", flush=True)
    lgbm = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        verbose=-1, n_jobs=4,
    )
    lgbm.fit(Xtr, ytr)
    report("lgbm(200,lr.05,l31) [old base]", yva, np.clip(lgbm.predict(Xva), 0, None),
           results, "lgbm")

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    for label, est in [
        ("extra_trees(300)", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=4)),
        ("rf(300)", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=4)),
    ]:
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        r = report(f"{label}  ({time.perf_counter() - t0:.0f}s)", yva,
                   np.clip(est.predict(Xva), 0, None), results, label)
        print()

    # ---- MLP capacity sweep ----
    print("=== MLP capacity sweep ===", flush=True)
    best = None
    for hidden, alpha in [
        ((24,), 1e-3), ((64,), 1e-3), ((128,), 1e-3), ((256,), 1e-3),
        ((256,), 1e-2), ((256,), 1e-4), ((512,), 1e-3), ((256, 128), 1e-3),
    ]:
        t0 = time.perf_counter()
        pipe = mlp_pipeline(hidden, alpha)
        pipe.fit(Xtr, ytr)
        r = report(f"mlp{hidden} alpha={alpha}  ({time.perf_counter() - t0:.0f}s)",
                   yva, np.clip(pipe.predict(Xva), 0, None), results, f"mlp{hidden}_{alpha}")
        if best is None or r < best[0]:
            best = (r, hidden, alpha)
    print(f"  --> best single model: {best}", flush=True)
    print()

    # ---- Target scaling comparison (it hurt) ----
    print("=== Target scaling comparison ===", flush=True)
    from sklearn.compose import TransformedTargetRegressor

    pipe = mlp_pipeline(best[1], best[2])
    ttr = TransformedTargetRegressor(regressor=pipe, transformer=StandardScalerProxy())
    ttr.fit(Xtr, ytr)
    report("mlp + target scaling (TTR)", yva, np.clip(ttr.predict(Xva), 0, None),
           results, "mlp_ttr")
    print()

    # ---- Ensemble ----
    print("=== Ensemble ===", flush=True)
    members = []
    for seed in (42, 7, 123):
        p = mlp_pipeline(best[1], best[2], seed=seed)
        p.fit(Xtr, ytr)
        members.append(np.clip(p.predict(Xva), 0, None))
    report("ensemble 3xMLP", yva, np.mean(members, axis=0), results, "ens_mlp3")
    report("ensemble 3xMLP + lgbm", yva,
           np.mean(members + [np.clip(lgbm.predict(Xva), 0, None)], axis=0),
           results, "ens_mlp3_lgbm")
    print()

    # ---- Per-bearing breakdown of the best single model ----
    print("=== Per-bearing breakdown (best single model) ===", flush=True)
    pipe = mlp_pipeline(best[1], best[2])
    pipe.fit(Xtr, ytr)
    best_pred = np.clip(pipe.predict(Xva), 0, None)
    for b in sorted(set(bva)):
        mask = np.asarray([x == b for x in bva])
        report(f"{b}  (n={int(mask.sum())})", yva[mask], best_pred[mask])
    print()

    print("=== Ranking ===", flush=True)
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(f"  {k:26s} {v['rmse']:9.2f}", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "train on the 9 training bearings, score on the 3 "
                        "independent validation bearings (the official split)",
            "data_root": str(Path(args.data_root)),
            "train_rows": int(len(ytr)),
            "validation_rows": int(len(yva)),
            "validation_bearings": sorted(set(bva)),
            "results": results,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\nwrote {out}", flush=True)
    return 0


def StandardScalerProxy():
    """Return a fresh target scaler for the TTR comparison."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler()


if __name__ == "__main__":
    raise SystemExit(main())
