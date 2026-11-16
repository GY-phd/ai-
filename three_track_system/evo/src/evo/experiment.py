"""实验记录层 — SwanLab 优先, 本地降级, 并做《实验记录规范》合规自检。

规范要点 (从赛题要求抽象):
  * 每个实验必须有 config / metrics / tags / notes
  * 需要能通过"链接或查询依据"定位到记录
  * 实验需可分组对比 (同一 project, 一致的命名与 tag 体系)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Optional

from .config import Config


# =====================================================================
# 本地记录器 (降级用)
# =====================================================================

class LocalRecorder:
    """写 JSONL 到本地, 保证离线环境下仍有完整可查记录。"""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "experiments.jsonl"

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("logged_at", time.time())
        # 写入前确保目录存在。
        # 只在 __init__ 建目录是不够的: 目录可能在运行期间被清理
        # (如上一轮实验产物被删), 或被多个并发运行共享。
        # 实测踩过: FileNotFoundError on experiments.jsonl。
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> List[Dict[str, Any]]:
        if not self.file.exists():
            return []
        out = []
        for line in self.file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


# =====================================================================
# 假设-实验-决策 循环 (赛题门槛②③)
# =====================================================================

@dataclass
class HypothesisRecord:
    """一次"读历史 → 提假设 → 做实验 → 做决策"的完整留痕。

    赛题门槛②③原文:
        ② "完整留存训练、调整、评估记录"
        ③ "至少3次'读历史→提假设→做实验→做决策'"

    因此每次迭代必须留下**四段式**证据链, 而不只是一个最终指标:
        read     —— 读到了什么历史 (漂移监测 + 归因 + 历史相似案例)
        hypothesis —— 提出什么可证伪的假设
        experiment —— 做了什么实验 (候选模型 + 影子评估 + 显著性)
        decision —— 做了什么决策 (上线/拒绝 + 理由)

    缺任何一段, 评委就无法验证"系统是真的在自主迭代"还是"只是在随机试参数"。
    """

    iteration: int
    # 1) 读历史
    read_history: Dict[str, Any] = field(default_factory=dict)
    # 2) 提假设 (必须可证伪: 声明预期效果与判定条件)
    hypothesis: str = ""
    expected_gain: float = 0.0
    falsification_criterion: str = ""
    # 3) 做实验
    experiment: Dict[str, Any] = field(default_factory=dict)
    # 4) 做决策
    decision: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    gain: float = 0.0
    cost_gpu_seconds: float = 0.0
    cost_labels: int = 0
    cost_tokens: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "read_history": self.read_history,
            "hypothesis": self.hypothesis,
            "expected_gain": self.expected_gain,
            "falsification_criterion": self.falsification_criterion,
            "experiment": self.experiment,
            "decision": self.decision,
            "approved": self.approved,
            "gain": round(self.gain, 6),
            "cost_gpu_seconds": round(self.cost_gpu_seconds, 4),
            "cost_labels": self.cost_labels,
            "cost_tokens": self.cost_tokens,
            "timestamp": self.timestamp,
        }

    def to_experiment_record(self) -> "ExperimentRecord":
        """转成统一的实验记录 (可写入 Swan Lab / 本地 JSONL)。"""
        return ExperimentRecord(
            name=f"iteration_{self.iteration:03d}_hypothesis_cycle",
            config={
                "hypothesis": self.hypothesis,
                "falsification_criterion": self.falsification_criterion,
                "expected_gain": self.expected_gain,
            },
            metrics={
                "gain": self.gain,
                "expected_gain": self.expected_gain,
                "approved": 1.0 if self.approved else 0.0,
                "cost_gpu_seconds": self.cost_gpu_seconds,
            },
            tags=["hypothesis_cycle", f"iter{self.iteration}"],
            notes=(
                f"读历史: {self.read_history.get('summary', '')} | "
                f"假设: {self.hypothesis} | "
                f"实验: {self.experiment.get('summary', '')} | "
                f"决策: {self.decision.get('reason', '')}"
            ),
            stage="hypothesis_cycle",
            iteration=self.iteration,
            cost_gpu_seconds=self.cost_gpu_seconds,
            cost_labels=self.cost_labels,
            cost_tokens=self.cost_tokens,
            extra=self.to_dict(),
        )


# =====================================================================
# 统一实验记录器
# =====================================================================

# 实验阶段 → 稳定的数值 id (供 SwanLab 作图时按阶段着色/筛选)
_STAGE_IDS: Dict[str, int] = {
    "bootstrap": 0,
    "iteration": 1,
    "retrain": 2,
    "hypothesis_cycle": 3,
    "gates": 4,
    "summary": 5,
    "batch": 6,
    "compliance": 7,
    "unknown": 8,
}


def _stage_id(stage: str) -> float:
    """阶段名 → 数值 id (未知阶段给一个稳定的派生值)。

    稳定性很重要: 不能用内置 `hash()` —— 它对 str 每进程随机加盐,
    会让不同运行之间的同一阶段 id 不同, 数据无法横向对比。
    """
    s = str(stage or "unknown")
    if s in _STAGE_IDS:
        return float(_STAGE_IDS[s])
    return float(sum(ord(c) for c in s) % 1000)


@dataclass
class ExperimentRecord:
    """一条实验记录 — 对齐《实验记录规范》要求的字段。"""

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    # 扩展字段 (规范未强制, 但有助于归因与成本分析)
    stage: str = "unknown"
    iteration: int = 0
    cost_gpu_seconds: float = 0.0
    cost_labels: int = 0
    cost_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "tags": self.tags,
            "notes": self.notes,
            "stage": self.stage,
            "iteration": self.iteration,
            "cost_gpu_seconds": self.cost_gpu_seconds,
            "cost_labels": self.cost_labels,
            "cost_tokens": self.cost_tokens,
            "extra": self.extra,
            "timestamp": self.timestamp,
        }


class ExperimentRecorder:
    """SwanLab 记录器, 无网络或无依赖时自动降级到本地 JSONL。

    **SwanLab API 适配** (对照 swanlab 0.10.x 的真实签名):

    ```python
    swanlab.init(*, workspace=None, project=None, public=None, name=None,
                 config=None, tags=None, log_dir=None, mode=None, ...)
    swanlab.log(data: Mapping[str, Any], step: Optional[int] = None)
    swanlab.finish()
    ```

    两个关键点 (早期实现踩过):
      1. 参数名是 **`name`**, 不是 `experiment_name`; 日志目录是 **`log_dir`**,
         不是 `logdir`。传错会 TypeError。
      2. **`log()` 接收扁平指标字典** —— `{"acc": 0.9, "loss": 0.1}`。
         把 `{"metrics": {...}, "config": {...}}` 整体塞进去, 只会得到
         一个叫 "metrics" 的怪指标, 图表上什么都看不到。正确做法是
         **把指标展平**, 并用 `step` 表达迭代序号, 这样曲线才有横轴。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.project = cfg.get("experiment.project", "zhixun-engine")
        self.workspace = cfg.get("experiment.workspace")
        self.local = LocalRecorder(cfg.path("experiment.local_dir"))
        self.mode = "local"
        self._run = None
        self._swan = None
        # Swan Lab 公开性是否已成功设置 (赛题硬要求: 记录公开)
        self._swan_public = False
        # 是否因版本不兼容而未能设置公开参数 (用于合规报告的诚实标注)
        self._public_set_failed = False
        # 已登录状态的缓存 (避免每次启动都重新登录)
        self._logged_in = False
        self._init_swanlab()

    # -- 初始化 ----------------------------------------------------------
    def _init_swanlab(self) -> None:
        """按配置决定是否启用 SwanLab。

        赛题评测环境无外联, 因此默认走本地 JSONL; 但**本地开发/提交材料
        准备**时需要真实上传到 SwanLab (赛题硬要求: 记录必须公开)。
        启用方式 (任一):
          * `runtime.offline: false`
          * 环境变量 `SWANLAB_ENABLE=1`
        """
        offline = bool(self.cfg.get("runtime.offline", True))
        enabled = os.environ.get("SWANLAB_ENABLE", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if offline and not enabled:
            self.mode = "local"
            return
        try:
            import swanlab  # type: ignore

            self._swan = swanlab
            self.mode = "swanlab"
        except Exception:
            self.mode = "local"

    # -- 登录 ------------------------------------------------------------
    def _ensure_login(self) -> bool:
        """确保已登录 SwanLab。

        凭据来源 (按优先级):
          1. 环境变量 `SWANLAB_API_KEY` —— **推荐**, 不落盘
          2. `swanlab.login()` 的交互式提示 —— 需要人工粘贴一次
          3. 本地已保存的凭据 (`swanlab.login(save=True)` 写入)

        返回是否成功登录。失败时不抛异常 —— 记录器会降级到本地 JSONL,
        闭环照常运行 (评测环境可能根本没网)。
        """
        if self._logged_in or self._swan is None:
            return self._logged_in
        try:
            api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
            if api_key:
                ok = self._swan.login(api_key=api_key, save=True)
            else:
                # 交互式登录: 若已有本地凭据会直接成功, 否则提示粘贴
                ok = self._swan.login(save=True)
            self._logged_in = bool(ok)
        except Exception:
            self._logged_in = False
        return self._logged_in

    def start_run(self, name: str, config: Dict[str, Any], tags: List[str] | None = None) -> None:
        """启动一次 SwanLab 实验。

        赛题硬性要求: **Swan Lab 记录必须设置为公开状态**, 评委才能访问。

        兼容性处理:
          * 参数名 `name` (新版) / `experiment_name` (旧版)
          * 日志目录 `log_dir` (新版) / `logdir` (旧版)
          * 公开参数 `public` (新版) / `visibility` / `mode`
            —— 逐个尝试; 全部失败则降级为不带公开参数启动, 并**诚实标记
            公开性未确认** (由 `compliance_report` 报出, 绝不谎报合规)。
        """
        if self.mode != "swanlab" or self._swan is None:
            return

        # 登录失败 → 降级到本地, 不阻断闭环
        if not self._ensure_login():
            self.mode = "local"
            return

        base: Dict[str, Any] = {
            "workspace": self.workspace,
            "project": self.project,
            "config": config or {},
            "tags": tags or [],
        }
        # 新版参数名
        base_variants: List[Dict[str, Any]] = [
            {"name": name, "log_dir": str(self.local.dir)},
            {"experiment_name": name, "logdir": str(self.local.dir)},
        ]
        # 公开参数变体 (参数名随版本不同)
        pub_variants: List[Dict[str, Any]] = []
        if bool(self.cfg.get("experiment.swanlab_public", True)):
            for pub in ({"public": True}, {"visibility": "public"}, {"mode": "public"}):
                for core in base_variants:
                    pub_variants.append({**core, **pub})

        # 先试带公开参数的组合, 再试不带的。
        #
        # 注意: 这里**必须先物化列表再遍历**。曾经写成
        #     for pub in pubs:
        #         for core in variants:      # ← 遍历的同时往 variants 追加
        #             variants.append(...)
        # 追加会让迭代器不断看到新元素 → 无限循环 → 实验进程挂死。
        # 这是真实踩过的坑 (表现为"测试永远不结束")。
        ordered: List[Tuple[Dict[str, Any], bool]] = (
            [(v, True) for v in pub_variants] + [(v, False) for v in base_variants]
        )
        for kwargs, is_public in ordered:
            try:
                self._run = self._swan.init(**{**base, **kwargs})
                self._swan_public = bool(is_public)
                self._public_set_failed = not is_public
                return
            except TypeError:
                continue
            except Exception:
                break
        # 全部失败 → 保持本地模式, 不崩
        self.mode = "local"
        self._swan_public = False
        self._public_set_failed = True


    # -- 写入 ------------------------------------------------------------
    def log(self, record: ExperimentRecord) -> None:
        """写入一条实验记录。

        **SwanLab 的 `log()` 接收扁平指标字典** —— 不是嵌套结构。
        正确用法: `swanlab.log({"acc": 0.9, "loss": 0.1}, step=3)`。

        这里做两件事:
          1. **展平**: 把 metrics 摊平成 `key -> value`, 并加上
             `stage`/`iteration`/`cost_*` 等标量, 让 SwanLab 图表上直接
             能画出性能曲线与成本曲线。
          2. **step**: 用迭代序号当横轴 —— 否则所有点挤在同一 step,
             曲线退化成一条水平线, 失去"随迭代变化"的可视化价值。

        非数值字段 (config/tags/notes) 不能进 log() —— SwanLab 只接受
        数值指标。它们保留在本地 JSONL 与 `run_summary.json` 里。
        """
        payload = record.to_dict()
        if self.mode == "swanlab" and self._swan is not None:
            flat: Dict[str, float] = {}
            for k, v in (record.metrics or {}).items():
                if isinstance(v, bool):
                    flat[k] = float(v)
                elif isinstance(v, (int, float)):
                    flat[k] = float(v)
            # 标量化的实验元信息 (全部是数值, 可作图)
            flat["stage_id"] = _stage_id(record.stage)
            flat["iteration"] = float(record.iteration)
            flat["cost_gpu_seconds"] = float(record.cost_gpu_seconds)
            flat["cost_labels"] = float(record.cost_labels)
            flat["cost_tokens"] = float(record.cost_tokens)
            if flat:
                try:
                    # step 用迭代序号; 无迭代信息时用 0
                    self._swan.log(flat, step=max(0, int(record.iteration)))
                except TypeError:
                    # 旧版不支持 step 参数
                    try:
                        self._swan.log(flat)
                    except Exception:
                        pass
                except Exception:
                    pass
        self.local.log(payload)

    def log_many(self, records: List[ExperimentRecord]) -> None:
        for r in records:
            self.log(r)

    def finish(self) -> None:
        if self.mode == "swanlab" and self._swan is not None:
            try:
                self._swan.finish()
            except Exception:
                pass

    # -- 合规自检 --------------------------------------------------------
    def compliance_report(self) -> Dict[str, Any]:
        """检查所有记录是否满足《实验记录规范》的必填字段要求。"""
        required = self.cfg.get("experiment.required_fields", []) or []
        records = self.local.load()
        missing: Dict[str, int] = {f: 0 for f in required}
        for rec in records:
            for field_name in required:
                if not rec.get(field_name):
                    missing[field_name] += 1
        n_missing = sum(missing.values())
        # Swan Lab 公开性检查 —— 赛题硬要求"记录必须设置为公开状态"
        swanlab_public = bool(self.cfg.get("experiment.swanlab_public", True))
        swanlab_ok = True
        swanlab_note = "未启用 SwanLab (本地 JSONL 降级), 公开性不适用"
        if self.mode == "swanlab":
            swanlab_ok = self._swan_public
            if self._swan_public:
                swanlab_note = "Swan Lab 实验已启动且已设置为公开"
            else:
                swanlab_note = (
                    "Swan Lab 实验已启动, 但当前版本未接受任何公开参数 "
                    "(public / visibility / mode) — 无法确认公开状态, "
                    "请手动到 Swan Lab 网页确认并设为 Public"
                )
        return {
            "mode": self.mode,
            "project": self.project,
            "workspace": self.workspace,
            "total_records": len(records),
            "required_fields": required,
            "missing_counts": missing,
            "compliant": len(records) > 0 and n_missing == 0 and swanlab_ok,
            "swanlab_public_required": swanlab_public,
            "swanlab_public_ok": swanlab_ok,
            "swanlab_note": swanlab_note,
            "local_log": str(self.local.file),
        }


# =====================================================================
# 运行级汇总
# =====================================================================

@dataclass
class RunSummary:
    """一次完整自进化运行的汇总, 用于报告与答辩。"""

    iterations: int = 0
    drifts_detected: int = 0
    updates_triggered: int = 0
    updates_approved: int = 0
    rollbacks: int = 0
    halted: bool = False
    total_gpu_seconds: float = 0.0
    total_labels: int = 0
    total_tokens: int = 0
    baseline_score: float = 0.0
    final_score: float = 0.0
    best_score: float = 0.0
    # 流上与冻结静态基线的对比 (长期稳定性)
    baseline_tail_score: float = 0.0
    tail_score: float = 0.0
    efficiency: float = 0.0
    # 多维度成本收益报告 (算力/标注/token/命中率)
    cost_benefit: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "drifts_detected": self.drifts_detected,
            "updates_triggered": self.updates_triggered,
            "updates_approved": self.updates_approved,
            "rollbacks": self.rollbacks,
            "halted": self.halted,
            "total_gpu_seconds": round(self.total_gpu_seconds, 3),
            "total_labels": self.total_labels,
            "total_tokens": self.total_tokens,
            "baseline_score": round(self.baseline_score, 5),
            "final_score": round(self.final_score, 5),
            "best_score": round(self.best_score, 5),
            "baseline_tail_score": round(self.baseline_tail_score, 5),
            "tail_score": round(self.tail_score, 5),
            "efficiency": round(self.efficiency, 10),
            "cost_benefit": self.cost_benefit,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        d = self.to_dict()
        d["records"] = self.records
        return d

    def to_compact_dict(self) -> Dict[str, Any]:
        """不含逐轮明细的摘要 (用于实验记录 metrics)。"""
        d = self.to_dict()
        d.pop("records", None)
        return d


def load_run_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
