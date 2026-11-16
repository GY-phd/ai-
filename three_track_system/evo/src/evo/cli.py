"""CLI 入口 — 赛题评测环境中的系统启动命令。

用法:
    python -m evo.cli run              # 运行自进化闭环
    python -m evo.cli run --synthetic  # 用合成数据跑通本地联调
    python -m evo.cli report           # 打印运行汇总与成本收益分析
    python -m evo.cli verify-audit     # 校验审计链完整性
    python -m evo.cli compliance       # 实验记录规范合规自检
    python -m evo.cli doctor           # 环境自检 (依赖/数据/服务)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import load_config
from .harness import Harness


def _apply_set(cfg, settings: list[str]) -> None:
    """应用 --set 覆盖 (dot-notation, 值尝试按 JSON 解析)。"""
    for kv in settings or []:
        k, _, v = kv.partition("=")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        cfg.set(k, val)


def _cfg_for(args: argparse.Namespace):
    """取配置 —— 优先用 main() 已应用 --set 的实例, 否则重新加载。"""
    cfg = getattr(args, "_cfg", None)
    if cfg is not None:
        return cfg
    return load_config(args.config)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _cfg_for(args)

    harness = Harness(cfg, use_synthetic=args.synthetic or args.force_synthetic)
    summary = harness.run()
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    cfg = _cfg_for(args)
    p = cfg.path("paths.artifacts_root") / "run_summary.json"
    if not p.exists():
        print("尚无运行结果, 请先执行 run", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    print("=== 运行汇总 ===")
    for k, v in data.items():
        if k not in ("records", "cost_benefit", "hypothesis_cycles", "policy_final"):
            print(f"  {k}: {v}")

    # 成本收益 (赛题客观分"资源效率" + 专家评审"成本收益")
    cb = data.get("cost_benefit")
    if cb:
        print("\n=== 成本收益 ===")
        print(f"  尾段增益: {cb.get('tail_gain')}")
        print(f"  每 GPU 秒增益: {cb.get('gain_per_gpu_second')}")
        print(f"  每标注配额增益: {cb.get('gain_per_label')}")
        print(f"  每千 token 增益: {cb.get('gain_per_1k_tokens')}")
        print(f"  触发命中率: {cb.get('trigger_approval_rate')}")
        res = cb.get("resources", {})
        print(
            f"  资源: 算力 {res.get('gpu_seconds')}s / 标注 {res.get('labels')} "
            f"/ token {res.get('llm_tokens')} / 迭代 {res.get('iterations')}"
        )

    # 假设-实验-决策 四段式 (赛题门槛②③)
    cycles = data.get("hypothesis_cycles") or []
    if cycles:
        print(f"\n=== 假设-实验-决策循环 ({len(cycles)} 次) ===")
        for h in cycles[:5]:
            print(
                f"  [{h['iteration']:03d}] 批准={h['approved']} "
                f"增益={h['gain']:.5f} | {h['hypothesis'][:60]}"
            )
        if len(cycles) > 5:
            print(f"  ... 共 {len(cycles)} 条 (详见 hypothesis_cycles.jsonl)")

    print("\n=== 每轮迭代 ===")
    for r in data.get("records", []):
        drift = r["drift"]
        print(
            f"  [{r['index']:03d}] psi={drift['max_psi']:.3f} "
            f"type={r['attribution'].get('drift_type'):<10} "
            f"action={r['decision'].get('action'):<16} "
            f"approved={r['approved']} gain={r['gain']:.5f} "
            f"gpu={r['cost_gpu_seconds']:.2f}s labels={r['cost_labels']}"
        )
    return 0


def _cmd_run_batches(args: argparse.Namespace) -> int:
    """批次下发模式 —— 赛题 B 包协议。

    流程: 取第 k 批 → 输出当前模型对 k 批的预测 + 上线决策 → 取第 k+1 批。
    每批的应答写入 `artifacts/batch_responses.jsonl`, 同时打印到 stdout
    (评测方可直接消费 stdout)。
    """
    from .batch import BatchRunner, CallbackBatchFetcher, DirectoryBatchFetcher
    from .data import OfficialAdapter

    cfg = _cfg_for(args)
    _apply_set(cfg, args.set)

    adapter = OfficialAdapter(cfg)
    if args.batch_dir:
        fetcher = DirectoryBatchFetcher(args.batch_dir, adapter)
    elif args.fetch_stdin:
        def _stdin_fetch(batch_id: int):
            # 协议: 评测方每批往 stdin 写一行 JSON {"X": [...], "y": [...]}
            line = sys.stdin.readline()
            if not line.strip():
                return None
            return json.loads(line)

        fetcher = CallbackBatchFetcher(_stdin_fetch, adapter)
    elif args.synthetic:
        # 合成模式: 直接把合成流按 batch_size 切批 —— 本地联调批次协议
        from .data import SyntheticStreamGenerator

        gen = SyntheticStreamGenerator(cfg)
        stream = gen.make_stream()
        size = int(cfg.get("data.batch_size", 500))

        def _syn_fetch(batch_id: int):
            lo = batch_id * size
            if lo >= len(stream):
                return None
            return stream.slice(lo, min(lo + size, len(stream)))

        fetcher = CallbackBatchFetcher(_syn_fetch, adapter)
    else:
        print(
            "批次模式需要 --batch-dir <目录>、--fetch-stdin 或 --synthetic",
            file=sys.stderr,
        )
        return 2

    runner = BatchRunner(cfg, fetcher, use_synthetic=args.synthetic)
    summary = runner.run(max_batches=args.max_batches)

    out = cfg.path("paths.artifacts_root") / "batch_responses.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in summary.responses:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_gates(args: argparse.Namespace) -> int:
    """打印基础运行门槛自检结果。"""
    cfg = _cfg_for(args)
    p = cfg.path("paths.artifacts_root") / "run_summary.json"
    if not p.exists():
        print("尚无运行结果, 请先执行 run", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    gates = data.get("gates")
    if not gates:
        print("run_summary.json 中无门槛记录 (该结果由旧版本生成)", file=sys.stderr)
        return 1
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    return 0 if gates.get("all_passed") else 2


def _cmd_verify_audit(args: argparse.Namespace) -> int:
    from .audit import AuditLog

    cfg = _cfg_for(args)
    log = AuditLog(cfg)
    res = log.verify()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("valid") else 2


def _cmd_compliance(args: argparse.Namespace) -> int:
    from .experiment import ExperimentRecorder

    cfg = _cfg_for(args)
    rec = ExperimentRecorder(cfg)
    res = rec.compliance_report()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("compliant") else 2


def _load_predict_model(cfg, checkpoint: str | None):
    """加载用于预测的模型。

    优先用显式 --checkpoint; 否则取运行产物里的现网模型 (registry 指针)。
    返回 (bundle, task, classes)。
    """
    import pickle
    from pathlib import Path

    from .data import OfficialAdapter

    task = str(cfg.get("data.task", "regression"))
    classes_cfg = cfg.get("data.classes")
    classes = [str(c) for c in classes_cfg] if classes_cfg else None

    if checkpoint:
        p = Path(checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"模型文件不存在: {p}")
        with p.open("rb") as fh:
            bundle = pickle.load(fh)
        return bundle, getattr(bundle, "task", task), classes

    # 找现网模型: 优先读 registry 指针 (models/current.txt), 再回退到最新版本
    artifacts = cfg.path("paths.artifacts_root")
    models_dir = artifacts / "models"
    pointer = models_dir / "current.txt"
    p: Path | None = None
    if pointer.exists():
        name = pointer.read_text(encoding="utf-8").strip()
        if name:
            cand = models_dir / name
            if cand.exists():
                p = cand
            else:
                for ext in (".pkl", ".joblib", ".bin"):
                    cand = models_dir / f"{name}{ext}"
                    if cand.exists():
                        p = cand
                        break
    if p is None and models_dir.exists():
        cands = sorted(
            models_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime
        )
        if cands:
            p = cands[-1]
    if p is None:
        raise FileNotFoundError(
            "未找到模型文件; 请先执行 run 生成模型, 或用 --checkpoint 指定"
        )
    with p.open("rb") as fh:
        bundle = pickle.load(fh)
    return bundle, getattr(bundle, "task", task), classes


def _cmd_serve(args: argparse.Namespace) -> int:
    """以官方 JSONL 协议接管 stdin/stdout 提供预测服务。

    评测平台的调用方式: 拉起容器 → 我们读一行请求 → 回一行响应。
    """
    from .predict_server import PredictServer

    cfg = _cfg_for(args)
    _apply_set(cfg, args.set)
    bundle, task, classes = _load_predict_model(cfg, args.checkpoint)
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    server = PredictServer(model=bundle, task=task, classes=classes)
    print(
        f"[evo] 预测服务启动: task={task} classes={classes} "
        f"model={getattr(bundle, 'name', '?')}",
        file=sys.stderr,
        flush=True,
    )
    return server.serve(sys.stdin, sys.stdout)


def _cmd_predict(args: argparse.Namespace) -> int:
    """离线批量预测 —— 本地校验输出格式是否符合官方协议。"""
    from .predict_server import predict_file

    cfg = _cfg_for(args)
    _apply_set(cfg, args.set)
    bundle, task, classes = _load_predict_model(cfg, args.checkpoint)
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    result = predict_file(
        bundle, task, args.input, args.output, classes=classes
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["n_errors"] == 0 else 2


def _cmd_swanlab(args: argparse.Namespace) -> int:
    """SwanLab 工具: 登录 / 检查状态 / 把已有运行产物补传上去。

    子命令:
      status   — 检查 swanlab 是否可用、是否已登录、项目/工作空间配置
      login    — 登录 (API Key 优先取环境变量 SWANLAB_API_KEY, 否则交互式)
      upload   — 把本地 experiments.jsonl 重新上传到 SwanLab (补传)
    """
    from .experiment import ExperimentRecorder

    cfg = _cfg_for(args)
    _apply_set(cfg, args.set)

    action = args.action
    # status / login / upload 必须**真正加载** swanlab —— 否则在默认的
    # offline 配置下永远查不到"是否已安装/已登录"。
    # 这不影响 run 命令的行为 (它仍按 runtime.offline 决定是否外联)。
    cfg.set("runtime.offline", False)
    recorder = ExperimentRecorder(cfg)

    if action == "status":
        info = {
            "swanlab_installed": recorder._swan is not None,
            "mode": recorder.mode,
            "project": recorder.project,
            "workspace": recorder.workspace,
            "public_required": bool(cfg.get("experiment.swanlab_public", True)),
            "public_confirmed": recorder._swan_public,
            "api_key_in_env": bool(os.environ.get("SWANLAB_API_KEY", "").strip()),
            "local_dir": str(recorder.local.file),
        }
        if recorder._swan is not None:
            try:
                info["swanlab_version"] = getattr(recorder._swan, "__version__", "unknown")
            except Exception:
                pass
            info["logged_in"] = recorder._ensure_login()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if info["swanlab_installed"] else 2

    if action == "login":
        if recorder._swan is None:
            print("未安装 swanlab, 请先 pip install swanlab", file=sys.stderr)
            return 2
        ok = recorder._ensure_login()
        print(json.dumps({"logged_in": ok}, ensure_ascii=False, indent=2))
        return 0 if ok else 2

    # upload: 把本地记录补传到 SwanLab
    if recorder._swan is None:
        print("未安装 swanlab", file=sys.stderr)
        return 2
    if not recorder._ensure_login():
        print("未登录 swanlab, 请先 `python -m evo swanlab login`", file=sys.stderr)
        return 2

    recorder.start_run(
        name=args.name or f"evo_upload_{time.strftime('%Y%m%d-%H%M%S')}",
        config={"source": "local_jsonl_upload"},
        tags=["zhixun-engine", "upload"],
    )
    records = recorder.local.load()
    for rec in records:
        recorder.log(
            ExperimentRecord(
                name=str(rec.get("name", "record")),
                config=rec.get("config") or {},
                metrics=rec.get("metrics") or {},
                tags=rec.get("tags") or [],
                notes=str(rec.get("notes", "")),
                stage=str(rec.get("stage", "unknown")),
                iteration=int(rec.get("iteration", 0) or 0),
                cost_gpu_seconds=float(rec.get("cost_gpu_seconds", 0.0) or 0.0),
                cost_labels=int(rec.get("cost_labels", 0) or 0),
                cost_tokens=int(rec.get("cost_tokens", 0) or 0),
            )
        )
    recorder.finish()
    print(json.dumps({"uploaded": len(records)}, ensure_ascii=False, indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg_for(args)
    checks: dict[str, object] = {}

    # Python 版本
    checks["python"] = sys.version.split()[0]

    # 关键依赖
    missing = []
    for mod in ("numpy", "pandas", "sklearn", "scipy", "lightgbm", "yaml"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    checks["missing_deps"] = missing

    # 数据
    root = cfg.path("paths.data_root")
    train = root / str(cfg.get("data.train_file"))
    valid = root / str(cfg.get("data.valid_file"))
    checks["data_root"] = str(root)
    checks["train_exists"] = train.exists()
    checks["valid_exists"] = valid.exists()

    # LLM 服务
    from .llm import LLMClient

    llm = LLMClient(cfg)
    checks["llm_available"] = llm.available()
    checks["llm_model"] = llm.model
    checks["llm_model_allowed"] = llm.model_allowed
    checks["llm_allowed_models"] = llm.allowed_models

    # ---- 赛题合规自检 ----
    checks["compliance"] = _compliance_checks(cfg, llm)
    violations = checks["compliance"]["violations"]

    checks["ok"] = not missing and not violations
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["ok"] else 2


def _compliance_checks(cfg, llm) -> dict:
    """一次性核对赛题的硬性约束 —— 提交前的合规体检。

    覆盖:
      * 仅允许千问 27B / 4B (平台限制)
      * 四核八G 无 GPU → 线程数必须受限
      * Swan Lab 记录必须公开
      * 提交包严禁含模型权重
      * 离线评测环境 → 不得外联
    """
    import os

    violations: list[str] = []
    notes: dict[str, object] = {}

    # 1) 模型白名单
    if not llm.model_allowed:
        violations.append(
            f"LLM 模型 {llm.model} 不在赛题允许的白名单 {llm.allowed_models} 内"
        )
    notes["llm_model_allowed"] = llm.model_allowed

    # 2) 线程数 (评测机仅 4 核)
    n_jobs = int(cfg.get("runtime.n_jobs", 4))
    if n_jobs <= 0 or n_jobs > 8:
        violations.append(f"runtime.n_jobs={n_jobs} 不合理 (评测环境为 4 核)")
    notes["n_jobs"] = n_jobs

    # 3) Swan Lab 公开性
    public_required = bool(cfg.get("experiment.swanlab_public", True))
    notes["swanlab_public_required"] = public_required
    if public_required:
        try:
            import swanlab  # type: ignore

            notes["swanlab_installed"] = True
        except Exception:
            notes["swanlab_installed"] = False
            violations.append("未安装 swanlab — 实验记录将降级为本地 JSONL, 无法满足公开记录要求")

    # 4) 权重不得进包
    try:
        import importlib.util
        import sys
        from pathlib import Path

        # scripts/ 不在包内, 用文件路径直接加载
        bs_path = Path(__file__).resolve().parents[2] / "scripts" / "build_submission.py"
        if bs_path.exists():
            spec = importlib.util.spec_from_file_location("_build_submission", bs_path)
            if spec and spec.loader:
                bs = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bs)
                leaked = [
                    p.name for p in bs._collect_files() if bs._is_weight(p)
                ]
                notes["weight_files_in_package"] = leaked
                if leaked:
                    violations.append(f"提交包中含权重文件: {leaked}")
        else:
            notes["weight_check_skipped"] = "未找到 scripts/build_submission.py"
    except Exception as exc:
        notes["weight_check_error"] = str(exc)

    # 5) 离线约束
    notes["offline"] = bool(cfg.get("runtime.offline", True))
    if not notes["offline"]:
        violations.append("runtime.offline=false — 评测环境无网络, 建议保持 true")

    # 6) 批次协议开关
    notes["batched"] = bool(cfg.get("data.batched", False))
    notes["batched_free_labels"] = bool(cfg.get("data.batched_free_labels", True))

    return {"violations": violations, "notes": notes}


def main(argv: list[str] | None = None) -> int:
    # 允许 --config 出现在子命令之后 (更符合直觉):
    #   python -m evo run --config configs/rul_bearing.yaml
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            config_value = argv[idx + 1]
            # 从子命令参数列表中移除, 提升为全局参数
            del argv[idx : idx + 2]
            argv = ["--config", config_value] + argv

    parser = argparse.ArgumentParser(prog="evo", description="智训引擎 — 工业模型自进化系统")
    parser.add_argument("--config", default=None, help="配置文件路径")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 所有子命令统一支持 --set 覆盖配置 (避免每个命令单独实现)
    def _add_set(p, help_text="覆盖配置, 如 --set data.task=regression"):
        p.add_argument("--set", action="append", default=[], help=help_text)
        return p

    p_run = _add_set(sub.add_parser("run", help="运行自进化闭环"))
    p_run.add_argument("--synthetic", action="store_true", help="强制使用合成数据")
    p_run.add_argument("--force-synthetic", action="store_true", help="同 --synthetic")
    p_run.set_defaults(func=_cmd_run)

    p_bat = _add_set(sub.add_parser("run-batches", help="批次下发模式 (赛题 B 包协议)"))
    p_bat.add_argument("--batch-dir", default=None, help="批次数据目录")
    p_bat.add_argument("--fetch-stdin", action="store_true", help="从 stdin 逐批读取 JSON")
    p_bat.add_argument("--max-batches", type=int, default=None, help="最大批次数")
    p_bat.add_argument("--synthetic", action="store_true", help="强制使用合成数据")
    p_bat.set_defaults(func=_cmd_run_batches)

    _add_set(sub.add_parser("report", help="打印运行汇总")).set_defaults(func=_cmd_report)
    _add_set(sub.add_parser("gates", help="基础运行门槛自检")).set_defaults(func=_cmd_gates)
    _add_set(sub.add_parser("verify-audit", help="校验审计链")).set_defaults(
        func=_cmd_verify_audit
    )
    _add_set(sub.add_parser("compliance", help="实验记录合规自检")).set_defaults(
        func=_cmd_compliance
    )
    _add_set(sub.add_parser("doctor", help="环境自检")).set_defaults(func=_cmd_doctor)

    # -- 官方评测协议: 预测服务与离线预测 --
    p_serve = _add_set(
        sub.add_parser("serve", help="以官方 JSONL 协议接管 stdin/stdout 提供预测服务")
    )
    p_serve.add_argument(
        "--checkpoint", default=None, help="模型文件路径 (默认取现网模型)"
    )
    p_serve.add_argument("--classes", default=None, help="分类任务的类别列表 (逗号分隔)")
    p_serve.set_defaults(func=_cmd_serve)

    p_pred = _add_set(sub.add_parser("predict", help="离线批量预测 (校验官方输出格式)"))
    p_pred.add_argument("--input", required=True, help="请求 JSONL 路径")
    p_pred.add_argument("--output", required=True, help="响应 JSONL 输出路径")
    p_pred.add_argument("--checkpoint", default=None, help="模型文件路径")
    p_pred.add_argument("--classes", default=None, help="分类任务的类别列表 (逗号分隔)")
    p_pred.set_defaults(func=_cmd_predict)

    # -- SwanLab 工具 --
    p_swan = _add_set(
        sub.add_parser("swanlab", help="SwanLab 记录工具 (登录/状态/补传)")
    )
    p_swan.add_argument(
        "action",
        choices=["status", "login", "upload"],
        help="status=检查状态, login=登录, upload=把本地记录补传到 SwanLab",
    )
    p_swan.add_argument("--name", default=None, help="upload 时的实验名")
    p_swan.set_defaults(func=_cmd_swanlab)

    args = parser.parse_args(argv)
    # --set 统一在此应用 (所有子命令都支持)
    if getattr(args, "set", None):
        cfg = _cfg_for(args)
        _apply_set(cfg, args.set)
        args._cfg = cfg
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
