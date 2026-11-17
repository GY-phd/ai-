"""数据层: 官方数据适配 + 合成漂移流。

设计要点:
  * 官方数据格式未知, 因此用一个 `OfficialAdapter` 把任意 CSV/Parquet
    归一化成内部统一 schema: 特征矩阵 X, 标签 y, 时间戳/批次 id。
  * 合成流仅用于本地联调与回归测试, 保证无网络也能端到端跑通闭环。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class Dataset:
    """统一数据容器。"""

    X: np.ndarray
    y: Optional[np.ndarray]
    meta: pd.DataFrame  # 含 timestamp / batch / 可选 group
    name: str = ""
    feature_names: list[str] = field(default_factory=list)
    # 官方 flatten 的字段路径 (schema)。官方 loader 会逐条比对, 不一致报错。
    # 随数据流传递到模型, 保证训练/推理用同一套 schema。
    feature_paths: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def has_labels(self) -> bool:
        """是否存在**可用**的标签。

        标签延迟场景下, 部分样本的标签尚未到达 (以 -1 标记),
        这类样本不能参与训练与评估, 但仍可用于推理与漂移监测。
        """
        if self.y is None or len(self.y) != len(self.X):
            return False
        return bool(np.any(self.y >= 0))

    @property
    def labeled_mask(self) -> np.ndarray:
        """返回"标签可用"的布尔掩码。"""
        if self.y is None or len(self.y) != len(self.X):
            return np.zeros(len(self.X), dtype=bool)
        return np.asarray(self.y) >= 0

    def labeled(self) -> "Dataset":
        """只保留标签可用的样本。"""
        if self.y is None or len(self.y) != len(self.X):
            return self
        m = self.labeled_mask
        if m.all():
            return self
        return Dataset(
            X=self.X[m],
            y=np.asarray(self.y)[m],
            meta=self.meta.iloc[m].reset_index(drop=True),
            name=f"{self.name}_labeled",
            feature_names=self.feature_names,
        )

    def slice(self, start: int, end: int) -> "Dataset":
        return Dataset(
            X=self.X[start:end],
            y=None if self.y is None else self.y[start:end],
            meta=self.meta.iloc[start:end].reset_index(drop=True),
            name=f"{self.name}[{start}:{end}]",
            feature_names=self.feature_names,
        )


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".feather",):
        return pd.read_feather(path)
    if suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    return pd.read_csv(path)


def _infer_label_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for cand in ("label", "y", "target", "fault", "class", "故障", "标签"):
        if cand in df.columns:
            return cand
    raise KeyError("无法识别标签列，请显式配置 data.label_col")


def _infer_time_col(df: pd.DataFrame) -> str | None:
    for cand in ("timestamp", "time", "ts", "datetime", "date", "cycle", "工况"):
        if cand in df.columns:
            return cand
    return None


class OfficialAdapter:
    """把官方数据包归一化为内部 schema。

    官方数据到位后, 只需保证其 CSV/Parquet 中包含特征列 + 标签列,
    若包含时间戳列则自动按时间排序 (漂移流依赖时间顺序)。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _build(self, df: pd.DataFrame, name: str) -> Dataset:
        label_col = _infer_label_col(df, self.cfg.get("data.label_col"))
        time_col = _infer_time_col(df)
        feature_cols = self.cfg.get("data.feature_cols") or [
            c for c in df.columns if c not in (label_col, time_col) and c != "batch"
        ]
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
        X = df[feature_cols].to_numpy(dtype=np.float64)
        y = df[label_col].to_numpy() if label_col in df.columns else None
        meta = pd.DataFrame(index=df.index)
        if time_col:
            meta["timestamp"] = df[time_col].to_numpy()
        else:
            # 无时间戳时以行号模拟递增时间, 保证流式顺序可复现
            meta["timestamp"] = np.arange(len(df), dtype=np.float64)
        meta["batch"] = self._assign_batches(meta)
        return Dataset(X=X, y=y, meta=meta, name=name, feature_names=list(feature_cols))

    @staticmethod
    def _assign_batches(meta: pd.DataFrame, size: int = 500) -> np.ndarray:
        """按固定间隔切批, 用于影子评估的分组统计。"""
        n = len(meta)
        return np.arange(n) // size

    # -- JSONL 赛题格式 (cpu-b-rul-* 包) --------------------------------
    #
    # 赛题 JSONL 每行结构:
    #   {"id": "...", "inputs": {...嵌套数值...}, "answer": 数值,
    #    "meta": {"bearing": "Bearing1_1", "snapshot": 1, ...}}
    #
    # 特征按"字段路径"确定性展开 (与官方 baselines/data.py 的 flatten 语义一致):
    # 字典按键名字典序遍历, 因此特征顺序完全可复现。
    def load_jsonl(self, path: str | Path, name: str) -> Dataset:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        ids: list[str] = []
        rows: list[list[float]] = []
        targets: list[float] = []
        bearings: list[str] = []
        snapshots: list[int] = []
        paths: Optional[list[str]] = None

        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "inputs" not in row:
                    raise ValueError(f"{path.name} 第 {lineno} 行缺少 inputs")
                cur_paths, values = _flatten_jsonl_inputs(row["inputs"])
                if paths is None:
                    paths = cur_paths
                elif cur_paths != paths:
                    raise ValueError(
                        f"{path.name} 第 {lineno} 行特征 schema 与首行不一致"
                    )
                ids.append(str(row.get("id", f"{name}_{lineno}")))
                rows.append(values)
                answer = row.get("answer")
                targets.append(float(answer) if answer is not None else -1.0)
                meta = row.get("meta") or {}
                bearings.append(str(meta.get("bearing", "unknown")))
                snapshots.append(int(meta.get("snapshot", lineno)))

        if not rows:
            raise ValueError(f"{path.name} 为空")

        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        # 特征名: 由路径生成可读名, 如 horizontal_kurtosis
        feature_names = _paths_to_names(paths or [])

        df = pd.DataFrame(index=range(len(X)))
        frame = pd.DataFrame(
            {
                "timestamp": np.arange(len(X), dtype=np.float64),
                "bearing": bearings,
                "snapshot": snapshots,
            }
        )
        frame["batch"] = OfficialAdapter._assign_batches(frame)
        return Dataset(
            X=X,
            y=y,
            meta=frame,
            name=name,
            feature_names=feature_names,
            # 保留官方 schema 的字段路径 —— 推理时逐条校验, 防止特征错位
            feature_paths=list(paths or []),
        )

    def load_jsonl_split(self, split: str) -> Dataset:
        p = self.cfg.path("paths.data_root") / self.cfg.get(f"data.{split}_file")
        return self.load_jsonl(p, split)

    def load_train(self) -> Dataset:
        # 赛题包使用 JSONL; 传统 CSV/Parquet 仍保留兼容
        fmt = str(self.cfg.get("data.format", "auto"))
        if fmt in ("jsonl", "auto"):
            p = self.cfg.path("paths.data_root") / self.cfg.get("data.train_file")
            if p.suffix in (".jsonl", ".json"):
                return self.load_jsonl(p, "train")
        return self._build(_read_table(p), "train")

    def load_valid(self) -> Dataset:
        fmt = str(self.cfg.get("data.format", "auto"))
        if fmt in ("jsonl", "auto"):
            p = self.cfg.path("paths.data_root") / self.cfg.get("data.valid_file")
            if p.suffix in (".jsonl", ".json"):
                return self.load_jsonl(p, "valid")
        return self._build(_read_table(p), "valid")

    def load_split(self, split: str) -> Dataset:
        p = self.cfg.path("paths.data_root") / f"{split}.csv"
        if not p.exists():
            p = self.cfg.path("paths.data_root") / f"{split}.parquet"
        return self._build(_read_table(p), split)


def _flatten_jsonl_inputs(inputs) -> tuple[list[str], list[float]]:
    """把嵌套 JSON 数值结构展开为 (路径列表, 数值列表)。

    语义与官方 baselines/data.py::flatten 保持一致:
      * 字典按键名排序遍历 → 特征顺序确定可复现
      * 数组按索引顺序展开
      * 目标派生字段 (rul/truerul 等) 会被拒绝, 防止标签泄漏
    """
    paths: list[str] = []
    values: list[float] = []

    def visit(value, path: list[tuple[str, str]], depth: int) -> None:
        if depth > 16:
            raise ValueError("输入嵌套层级过深")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ValueError("输入键名必须是字符串")
                if _normalize_key(key) in _FORBIDDEN_INPUTS:
                    raise ValueError(f"禁止把目标派生字段作为输入: {key}")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", str(index))], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # 与官方 baselines/data.py::flatten 语义对齐:
            # 必须有限 (官方用 math.isfinite 过滤 NaN/Inf)
            if not math.isfinite(float(value)):
                raise ValueError("模型输入只允许有限数值特征")
            paths.append(
                json.dumps(path, ensure_ascii=False, separators=(",", ":"))
            )
            values.append(float(value))
        else:
            raise ValueError("模型输入只允许有限数值特征")

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("期望 1..4096 个数值特征")
    return paths, values


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


_FORBIDDEN_INPUTS = {
    "truerul",
    "normalizedrul",
    "remainingusefullife",
    "rul",
    "totalsnapshots",
}


def _paths_to_names(paths: list[str]) -> list[str]:
    """把 JSON 路径转成可读特征名, 如 horizontal_kurtosis。"""
    names: list[str] = []
    for p in paths:
        try:
            parsed = json.loads(p)
        except Exception:
            names.append(p)
            continue
        parts = [
            seg[1] if isinstance(seg, list) and len(seg) == 2 and seg[0] == "key" else "idx"
            for seg in parsed
        ]
        names.append("_".join(str(x) for x in parts) or "f")
    return names


# =====================================================================
# 轴承生命周期漂移流 — 把多根轴承的完整生命周期拼成一条流
# =====================================================================

def build_bearing_stream(
    datasets: list[Dataset],
    cfg: Config,
    name: str = "bearing_stream",
) -> Dataset:
    """把多根轴承的完整生命周期拼接成一条漂移流。

    **为什么这样构造流 (这是赛题 B 的核心难点)**:

    赛题 B 是轴承剩余寿命回归, 训练/验证/隐藏测试按**整根轴承**隔离。
    真实工业现场是: 设备持续运行 → 振动信号随退化漂移 → 需要持续重训。
    隐藏测试流里的每根轴承都是"从健康跑到失效"的完整生命周期。

    因此把多根轴承的生命周期按顺序拼接成流:
      * 每根轴承内部按 snapshot 递增 (时间顺序)
      * 轴承之间形成**分布切换** (不同工况/退化速率 → 真实协变量漂移)

    这样得到的流同时具备:
      1. 轴承内的时间演化 (概念漂移: 退化进程)
      2. 轴承间的分布切换 (协变量漂移)

    meta 中保留 bearing/snapshot, 便于按轴承做分组评估与泄漏检查。
    """
    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_meta: list[pd.DataFrame] = []

    offset = 0
    for ds in datasets:
        if ds.X is None or len(ds.X) == 0:
            continue
        parts_X.append(ds.X)
        parts_y.append(ds.y if ds.y is not None else np.full(len(ds.X), -1.0))
        m = ds.meta.reset_index(drop=True).copy()
        # 时间戳在全流内全局递增, 保证流式顺序单调
        m["timestamp"] = np.arange(len(ds)) + offset
        offset += len(ds)
        parts_meta.append(m)

    if not parts_X:
        raise ValueError("无法构造漂移流: 所有数据集均为空")

    X = np.vstack(parts_X)
    y = np.concatenate(parts_y)
    meta = pd.concat(parts_meta, ignore_index=True)
    meta["batch"] = np.arange(len(meta)) // 500
    feature_names = list(datasets[0].feature_names)
    return Dataset(X=X, y=y, meta=meta, name=name, feature_names=feature_names)


def group_by_bearing(ds: Dataset) -> list[Dataset]:
    """把一个数据集按 bearing 列拆成多个子集 (各自是完整生命周期)。"""
    if "bearing" not in ds.meta.columns:
        return [ds]
    out: list[Dataset] = []
    for bearing, idx in ds.meta.groupby("bearing").groups.items():
        index = np.asarray(sorted(idx))
        m = ds.meta.iloc[index].reset_index(drop=True)
        out.append(
            Dataset(
                X=ds.X[index],
                y=None if ds.y is None else ds.y[index],
                meta=m,
                name=f"{ds.name}_{bearing}",
                feature_names=list(ds.feature_names),
            )
        )
    return out


# =====================================================================
# 合成漂移流生成器 — 本地联调 & CI 回归用
# =====================================================================

@dataclass
class DriftEvent:
    """一次漂移事件定义。"""

    at: float  # 0~1 流位置
    kind: str  # covariate | concept | mixed
    strength: float


class SyntheticStreamGenerator:
    """生成带渐进式漂移的二分类数据流。

    漂移类型:
      * covariate : 特征分布平移 (输入分布漂移)
      * concept   : 标签与特征关系改变 (输入-输出关系漂移)
      * mixed     : 两者同时
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        sc = cfg.section("data.synthetic")
        self.n_features = int(sc.get("n_features", 20))
        # 合成流的随机源必须**跟随运行种子** (runtime.seed), 否则多个随机种子
        # 会跑出完全相同的数据 —— 对照实验里的 "N 个种子取均值±std" 变成
        # 同一次结果重复 N 遍, 方差恒为 0, 无法反映系统的真实稳定性。
        # 优先用 runtime.seed; 未显式设置时才回落到 data.synthetic.random_state。
        runtime_seed = cfg.get("runtime.seed")
        base_state = int(sc.get("random_state", 42))
        self.random_state = int(runtime_seed) if runtime_seed is not None else base_state
        self.drift_points = list(sc.get("drift_points", [0.25, 0.55]))
        self.strength = float(sc.get("drift_strength", 0.3))
        # 复杂度演化: 初始噪声与噪声增长率
        # (噪声上升 → 最优超参随时间移动 → 策略层才有真实价值)
        self.base_noise = float(sc.get("base_noise", 0.2))
        self.noise_growth = float(sc.get("noise_growth", 1.6))
        self._rng = np.random.default_rng(self.random_state)
        self._w_true = self._rng.normal(size=self.n_features)
        self._w_true[:3] *= 3.0  # 前几个特征是强信号

    # -- 内部 ------------------------------------------------------------
    def _sample_block(self, n: int, prog: float) -> Tuple[np.ndarray, np.ndarray]:
        """按漂移进度 prog∈[0,1] 采样一个块。

        漂移维度:
          1. 协变量漂移  — 特征分布平移 (输入分布变了)
          2. 概念漂移    — 真权重方向改变 (输入-输出关系变了)
          3. 复杂度演化  — 噪声水平随时间上升 (数据变得更难拟合)

        第 3 点对策略层至关重要: 当数据难度变化时, **最优超参会随之变化**
        (噪声大时需要更强正则/更多叶节点, 噪声小时需要更简单的模型)。
        这正是策略层自改进的真实用武之地 —— 若最优超参恒定,
        任何超参搜索都只能是噪声。
        """
        rng = np.random.default_rng(self.random_state + int(prog * 1000))
        mu = np.zeros(self.n_features)
        shift = np.zeros(self.n_features)
        # 每 1/3 概率的漂移点改变一半特征均值
        for i, dp in enumerate(self.drift_points):
            if prog >= dp:
                strength = self.strength * min(1.0, (prog - dp) * 4 + 0.5)
                half = self.n_features // 2
                idx = slice(0, half) if i % 2 == 0 else slice(half, self.n_features)
                shift[idx] = strength * rng.normal(size=idx.stop - idx.start)
        # 协变量漂移: 改变输入分布
        X = rng.normal(loc=mu + shift, size=(n, self.n_features))

        # 概念漂移: 逐步改变真权重方向
        w = self._w_true.copy()
        concept_prog = 0.0
        for dp in self.drift_points:
            if prog >= dp:
                concept_prog = max(concept_prog, min(1.0, (prog - dp) * 3))
        if concept_prog > 0:
            rng2 = np.random.default_rng(self.random_state + 77)
            w_new = rng2.normal(size=self.n_features)
            w = (1 - concept_prog) * self._w_true + concept_prog * w_new

        # 复杂度演化: 噪声随流推进而上升 (最优点随之移动)
        noise = self.base_noise + self.noise_growth * prog

        # 非线性交互: 让浅树/线性模型无法完美拟合, 制造可区分的超参差异
        # (工业数据普遍含交互项与阈值效应; 若数据过于可分, 任何超参得分都是 1.0,
        #  超参搜索就失去意义 —— 这正是本合成器必须避免的失真)
        X = self._add_interactions(X, rng)

        # 交互项扩展了特征维度, 真权重需同步扩展
        if X.shape[1] != len(w):
            wr = np.random.default_rng(self.random_state + 999)
            w = np.concatenate([w, wr.normal(scale=0.8, size=X.shape[1] - len(w))])

        logit = X @ w + rng.normal(scale=noise, size=n)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n) < p).astype(np.int64)

        # 标签噪声: 工业现场的标注本身就有误差。加入可控的标签翻转后,
        # 任务不再可被任何超参完美拟合 (score 饱和到 1.0), 浅树与深树
        # 才会因"拟合噪声 vs 泛化"而真正分化 —— 策略层的价值空间来自这里。
        flip = float(self.cfg.get("data.synthetic.label_noise", 0.0) or 0.0)
        if flip > 0:
            flip_mask = rng.random(n) < flip
            y = np.where(flip_mask, 1 - y, y).astype(np.int64)
        return X, y

    def _add_interactions(self, X: np.ndarray, rng) -> np.ndarray:
        """在特征上叠加交互项与阈值效应, 提高数据难度。

        关键: 每个交互项**保留为独立特征**, 不能压缩成单列。
        早期版本用 `inter_mat @ weights` 把全部交互压成一个标量列 ——
        这既丢失了交互信息, 又让特征几乎线性可分, 结果是**任何超参组合
        都饱和到 score=1.0**, 彼此无法区分。后果是策略层的超参搜索
        只能在噪声里打转 (对照实验里完整系统花 66% 更多算力却追平固定超参)。

        保留独立交互列后, 浅树 (num_leaves 小) 与深树 (num_leaves 大)
        才会真正分化 —— 这正是策略层自改进的前提。
        """
        n, d = X.shape
        inter = []
        n_inter = max(2, d // 4)
        for _ in range(n_inter):
            i, j = rng.choice(d, size=2, replace=False)
            inter.append(X[:, i] * X[:, j])
        # 阈值效应: 某些特征越过阈值后才有影响
        for _ in range(max(1, d // 5)):
            i = rng.integers(d)
            thr = float(rng.normal())
            inter.append(np.maximum(0.0, X[:, i] - thr))
        if not inter:
            return X
        inter_mat = np.column_stack(inter)
        # 交互项自身的权重按固定种子生成, 保证可复现
        # (交互列现在直接进入 X, 权重在 _sample_block 里统一扩展到 w)
        wr = np.random.default_rng(self.random_state + 999)
        _ = wr  # 权重生成逻辑已上移到 _sample_block, 此处保留可复现锚点
        return np.column_stack([X, inter_mat])

    # -- 对外 ------------------------------------------------------------
    def make_train_valid(self) -> Tuple[Dataset, Dataset]:
        sc = self.cfg.section("data.synthetic")
        n_train = int(sc.get("n_train", 4000))
        n_valid = int(sc.get("n_valid", 1000))
        X, y = self._sample_block(n_train, 0.0)
        meta = pd.DataFrame(
            {
                "timestamp": np.arange(n_train, dtype=np.float64),
                "batch": np.arange(n_train) // 500,
            }
        )
        train = Dataset(X, y, meta, "synthetic_train", [f"f{i}" for i in range(self.n_features)])
        X, y = self._sample_block(n_valid, 0.0)
        meta = pd.DataFrame(
            {
                "timestamp": np.arange(n_valid, dtype=np.float64),
                "batch": np.arange(n_valid) // 500,
            }
        )
        valid = Dataset(X, y, meta, "synthetic_valid", [f"f{i}" for i in range(self.n_features)])
        return train, valid

    def make_stream(self) -> Dataset:
        sc = self.cfg.section("data.synthetic")
        n = int(sc.get("n_stream", 6000))
        chunks = 20
        size = n // chunks
        Xs, ys, ts = [], [], []
        for c in range(chunks):
            prog = c / (chunks - 1)
            X, y = self._sample_block(size, prog)
            Xs.append(X)
            ys.append(y)
            ts.append(np.full(size, c))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        ts = np.concatenate(ts)
        meta = pd.DataFrame(
            {"timestamp": np.arange(len(X), dtype=np.float64), "batch": np.arange(len(X)) // 500}
        )
        return Dataset(X, y, meta, "synthetic_stream", [f"f{i}" for i in range(self.n_features)])


def dataset_fingerprint(ds: Dataset) -> str:
    """数据集指纹, 用于实验记录的可追溯性。"""
    h = hashlib.sha256()
    h.update(str(ds.X.shape).encode())
    h.update(np.ascontiguousarray(ds.X[: min(256, len(ds.X))]).tobytes())
    if ds.has_labels:
        h.update(np.ascontiguousarray(ds.y[: min(256, len(ds.y))]).tobytes())
    return h.hexdigest()[:16]


def iter_windows(
    stream: Dataset, window: int, step: int
) -> Iterator[Tuple[int, int, Dataset]]:
    """按滑窗切流, 产出 (start, end, window_dataset)。"""
    for start in range(0, max(1, len(stream) - window + 1), step):
        end = min(start + window, len(stream))
        yield start, end, stream.slice(start, end)
