"""模型层: 候选模型族、训练、评估、影子对比、版本管理。

支持:
  * lgbm   —— LightGBM (主路线, CPU 友好)
  * logreg —— 逻辑回归 (强基线, 零超参, 极低算力成本)
  * rf     —— 随机森林
  * mlp    —— 小型 MLP (torch 可选, 缺失时自动跳过)

所有模型统一封装为 `ModelBundle`, 便于保存/加载/推理。
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .data import Dataset


# =====================================================================
# 指标
# =====================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str = "binary") -> Dict[str, float]:
    """计算评测指标。

    分类任务给出 AUC/F1/ACC/PREC/REC; 回归任务给出 RMSE/MAE/R²。

    **score 统一为"越大越好"**:
      * 分类: 0.7*AUC + 0.3*F1
      * 回归: -RMSE (取负 RMSE 作为越大越好的可比分数)
        这样影子评估、增益计算等逻辑在两种任务下语义一致。
    """
    out: Dict[str, float] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ---- 过滤"标签未到达"的样本 ----
    #
    # 标签延迟场景用 -1 标记未到达。**但官方 A 包的 answer 是英文类别名**
    # (Ethanol / Ethylene / ...), 不是数值 —— 对字符串数组做 `>= 0` 会直接抛
    # `TypeError`/`UFuncNoLoopError`。因此必须按 dtype 分派:
    #   * 数值标签 → 用 >= 0 过滤 (RUL 与整数编码分类)
    #   * 字符串标签 → 不过滤 (类别名不可能表示"标签未到达")
    if np.issubdtype(y_true.dtype, np.number):
        mask = y_true >= 0
        if not mask.all():
            y_true = y_true[mask]
            y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return (
            {k: 0.0 for k in ("acc", "f1", "precision", "recall", "auc", "score")}
            if task == "binary"
            else {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "score": 0.0}
        )
    if task == "binary" or task == "multiclass":
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # ---- 区分二分类与多分类 ----
        #
        # 官方 A 包是 **6 类气体分类** (Ethanol/Ethylene/Ammonia/Acetaldehyde/
        # Acetone/Toluene), answer 是**英文类别名**。
        # sklearn 的 precision/recall/f1 默认 average='binary' —— 对多分类
        # 会直接抛 ValueError, 且 AUC 也需要概率输入而非标签。
        # 因此按实际类别数分派:
        #   * 2 类 → binary 口径 (含 AUC)
        #   * >2 类 → macro 口径 (对齐官方 Macro-F1 主指标)
        try:
            n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        except Exception:
            n_classes = 2
        is_multi = n_classes > 2

        # A judge slice can contain a single class - a short stream window
        # of one gas, or one engine's last few cycles.  sklearn's binary
        # path assumes label 1 is the positive class and raises
        # ``ValueError: pos_label=1 is not a valid label`` when it is
        # not present.  That is a crash on a legitimate measurement.
        #
        # The same crash happens with *two* classes when neither of them
        # is the label 1 that sklearn's binary path assumes.  The labels
        # in this system are the integers 0..5 of the six gases, so a
        # slice holding only, say, Acetaldehyde (3) and Toluene (5) has
        # two classes and no 1 - and binary averaging is meaningless
        # for them anyway, because "which of these two is the positive
        # class" has no answer.
        #
        # So the binary path is used only when the slice really is a
        # 0/1 problem; everything else goes through the macro path.
        labels_present = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = (len(labels_present) == 2
                     and set(labels_present.tolist()) == {0, 1})
        avg = "binary" if is_binary else "macro"
        is_multi = not is_binary
        out["acc"] = float(accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        out["precision"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        out["recall"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        out["n_classes"] = int(n_classes)
        if is_multi:
            # 多分类 AUC 需要 one-hot 概率; 这里用标签的 one-hot 近似
            # (sklearn 支持 label_binarize 风格的输入)。
            try:
                classes = np.unique(np.concatenate([y_true, y_pred]))
                if len(classes) < 2:
                    raise ValueError("AUC needs at least two classes")
                yt = np.searchsorted(classes, y_true)
                yp = np.searchsorted(classes, y_pred)
                out["auc"] = float(
                    roc_auc_score(yt, yp, multi_class="ovr", average="macro")
                )
            except Exception:
                out["auc"] = 0.5
        else:
            try:
                out["auc"] = float(roc_auc_score(y_true, y_pred))
            except Exception:
                out["auc"] = 0.5
        # 复合得分: AUC 为主, F1 为辅 (与工业误报/漏报成本对齐)
        out["score"] = 0.7 * out["auc"] + 0.3 * out["f1"]
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
        out["r2"] = float(r2_score(y_true, y_pred))
        # score = -RMSE: 直接对齐赛题主指标 (RMSE 越低越好 → score 越大越好),
        # 且增量的物理单位是"分钟", 使影子评估的 margin 有明确含义。
        out["score"] = -out["rmse"]
    return out


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


# =====================================================================
# 模型封装
# =====================================================================

@dataclass
class ModelBundle:
    """模型 + 元信息。"""

    name: str
    estimator: Any
    feature_names: List[str]
    task: str = "binary"
    metrics: Dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    train_size: int = 0
    version: str = ""
    notes: str = ""
    # 特征 schema 的**字段路径** (官方 baselines/data.py::flatten 的 paths)。
    # 官方 loader 会逐条比对 paths, 不一致直接报错。
    # 这里随模型一起保存, 推理时严格校验 —— 避免"静默错位"这类最难查的 bug。
    feature_paths: List[str] = field(default_factory=list)

    # -- 推理 ------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba") and self.task == "binary":
            p = self.estimator.predict_proba(X)[:, 1]
            return (p >= 0.5).astype(np.int64)
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        return self.predict(X).astype(np.float64)

    # -- 持久化 ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh)
        meta = {
            "name": self.name,
            "task": self.task,
            "metrics": self.metrics,
            "train_seconds": self.train_seconds,
            "params": self.params,
            "train_size": self.train_size,
            "version": self.version,
            "feature_names": self.feature_names,
            "notes": self.notes,
        }
        p.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return p

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


# =====================================================================
# 训练器
# =====================================================================

class ModelFactory:
    """按名称构建候选模型 —— 自动适配分类/回归任务。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.task = str(cfg.get("data.task", "binary"))
        # 评测环境: 四核八G Linux, **无 GPU**。
        # 因此所有模型必须限制线程数 —— 否则 LightGBM 默认吃满全部核,
        # 在评测机上与其他任务争抢 CPU, 既拖慢自身也违背"资源效率"评分。
        self.n_jobs = int(cfg.get("runtime.n_jobs", 4))

    @property
    def is_regression(self) -> bool:
        return self.task == "regression"

    def _thread_params(self, params: Dict[str, Any], name: str) -> Dict[str, Any]:
        """注入线程数限制 (评测环境仅 4 核)。

        注意: sklearn 1.8 起 `LogisticRegression` 的 `n_jobs` 已废弃且无效,
        传了会告警。因此只对**真正支持 n_jobs** 的模型注入。
        """
        if name in ("lgbm", "rf"):
            params.setdefault("n_jobs", self.n_jobs)
        # random_state 统一固定以保证可复现
        params.setdefault("random_state", int(self.cfg.get("runtime.seed", 42)))
        return params

    def _scaled(self, est: Any, params: Dict[str, Any]) -> Any:
        """把模型包进标准化 Pipeline。

        缩放器与模型**绑定成一个对象**一起 fit / 持久化 —— 这样:
          * scaler 只在该候选的训练集上拟合 (验证集/留出集绝不参与);
          * 推理时自动复用训练时拟合的均值/方差, 不会每次重新计算;
          * 不会出现"训练时缩放过、推理时忘了缩放"这类静默错位。

        **关于目标缩放 (重要实测结论)**:
            早期实现用 `TransformedTargetRegressor` 把目标也标准化。
            在赛题 B (XJTU-SY 轴承 RUL) 上实测这是**明显有害**的:

                TransformedTargetRegressor 包裹 -> RMSE 716.03
                仅输入标准化 (目标保持原始分钟数) -> RMSE 521.07   (好 27%)

            原因: RUL 是 (总快照数 - 当前快照) 的**结构性派生量**, 尺度本身
            携带信息 (长寿命轴承的 RUL 量级是短寿命轴承的 20 倍)。目标
            标准化把所有轴承压到同一个 z-score 尺度, 等于抹掉了"这根轴承
            能活多久"这一维信息, 网络再用隐层去重建它极其困难。

            树模型 (lgbm / rf) 对目标尺度不敏感, 原本没问题; 但 MLP 作为
            基线后这个包装就成了拖累, 且它还引入了 sklearn 的
            `process_routing` bug (传 sample_weight 时报 KeyError: 'regressor')。

            因此这里**只做输入标准化**, 目标保持原始单位。
            若某个场景确实需要目标缩放, 可通过 `model.target_scaling` 开关
            显式打开 (默认关闭)。
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps: list[tuple[str, Any]] = [
            ("scaler", StandardScaler()),
            ("model", est),
        ]
        pipe = Pipeline(steps)

        if self.is_regression and bool(
            self.cfg.get("model.target_scaling", False)
        ):
            from sklearn.compose import TransformedTargetRegressor

            # 目标缩放: 标准化后再回归, 预测自动还原为"分钟/循环数"
            ttr = TransformedTargetRegressor(
                regressor=pipe, transformer=StandardScaler()
            )
            return ttr
        return pipe

    def build(self, name: str, overrides: Dict[str, Any] | None = None) -> Any:
        params = dict(self.cfg.get(f"model.{name}_params", {}) or {})
        if overrides:
            params.update(overrides)
        reg = self.is_regression
        params = self._thread_params(params, name)

        if name == "lgbm":
            if reg:
                from lightgbm import LGBMRegressor

                params.setdefault("objective", "regression")
                params.setdefault("verbose", -1)
                # 早停的 rounds 只在 Trainer 传入 eval_set 时才由 callbacks 生效;
                # 这里必须移除它, 否则 LightGBM 会自行开启早停却拿不到评估集。
                params.pop("early_stopping_rounds", None)
                params.pop("early_stopping", None)
                return LGBMRegressor(**params)
            from lightgbm import LGBMClassifier

            params.setdefault("verbose", -1)
            params.pop("early_stopping_rounds", None)
            params.pop("early_stopping", None)
            return LGBMClassifier(**params)

        if name == "logreg":
            if reg:
                from sklearn.linear_model import Ridge

                # 线性模型同样需要输入标准化 (RUL 特征量级悬殊),
                # 目标也做缩放以改善收敛条件。
                return self._scaled(Ridge(**params), params)
            from sklearn.linear_model import LogisticRegression

            params.setdefault("max_iter", 1000)
            return LogisticRegression(**params)

        if name == "rf":
            if reg:
                from sklearn.ensemble import RandomForestRegressor

                params.setdefault("n_estimators", 200)
                params.setdefault("random_state", 42)
                return RandomForestRegressor(**params)
            from sklearn.ensemble import RandomForestClassifier

            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", 42)
            return RandomForestClassifier(**params)

        if name == "mlp":
            # MLP 对输入尺度极其敏感 —— RUL 任务的特征量级可差几个数量级
            # (振动能量 vs 频谱质心), 不缩放会直接导致收敛失败。
            #
            # 官方基线的做法: "输入均值、标准差及回归目标缩放最初只在训练集拟合;
            # 续训沿用检查点状态。验证集和隐藏集不能参与拟合这些统计量。"
            #
            # 这里用 sklearn Pipeline 把 StandardScaler 与模型绑在一起 ——
            # scaler **只在该候选自己的训练集上拟合**, 验证集/留出集绝不参与,
            # 且整条 Pipeline 一起持久化, 推理时用训练时拟合的状态。
            if reg:
                from sklearn.neural_network import MLPRegressor

                params.setdefault("hidden_layer_sizes", (64, 32))
                params.setdefault("max_iter", 300)
                params.setdefault("random_state", 42)
                return self._scaled(MLPRegressor(**params), params)
            from sklearn.neural_network import MLPClassifier

            params.setdefault("hidden_layer_sizes", (64, 32))
            params.setdefault("max_iter", 300)
            params.setdefault("random_state", 42)
            return self._scaled(MLPClassifier(**params), params)

        if name == "extra_trees":
            if reg:
                from sklearn.ensemble import ExtraTreesRegressor

                params.setdefault("n_estimators", 200)
                params.setdefault("random_state", 42)
                return ExtraTreesRegressor(**params)
            from sklearn.ensemble import ExtraTreesClassifier

            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", 42)
            return ExtraTreesClassifier(**params)

        raise ValueError(f"未知模型: {name}")


class Trainer:
    """训练候选模型并记录算力开销。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.factory = ModelFactory(cfg)
        self.task = cfg.get("data.task", "binary")

    @staticmethod
    def _weight_fit_kwargs(
        est: Any, sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """把样本权重路由到真正接受它的那个层。

        sklearn 的 `Pipeline.fit` **不接受** `sample_weight` 这个 fit 参数,
        必须写成 `stepname__sample_weight`。而回归候选 (mlp / logreg) 在
        `ModelFactory._scaled()` 里被包成

            TransformedTargetRegressor(
                regressor=Pipeline([("scaler", ...), ("model", est)])
            )

        所以有两层前缀:
          * TransformedTargetRegressor -> `regressor__sample_weight`
          * Pipeline                   -> `model__sample_weight`

        裸模型 (lgbm / rf / extra_trees) 不需要任何前缀。

        路由错误的后果不是"警告"而是**直接崩**:
            ValueError: Pipeline.fit does not accept the sample_weight parameter
        这个坑在 lgbm 作为基线时不会暴露 (它没有 Pipeline),
        换成 mlp 作为基线后第一次重训就炸 —— 属于典型的"换模型才暴露"的
        隐患, 因此在这里统一处理, 而不是让每个模型各自适配。

        注: 若显式开启 `model.target_scaling`, 外层会多一层
        `TransformedTargetRegressor` —— 但当前 sklearn 版本的
        `process_routing` 对它有 bug (传权重时 KeyError: 'regressor'),
        故该组合下**放弃权重**而不是崩掉整个训练流程。
        """
        if sample_weight is None:
            return {}

        # TransformedTargetRegressor: 不再传权重 (sklearn routing bug),
        # 宁可少一层"近端样本加权", 也不能让训练直接失败。
        if est.__class__.__name__ == "TransformedTargetRegressor":
            return {}

        # Pipeline: 权重交给最后一步 (真正的估计器)
        if hasattr(est, "steps"):
            last = list(est.steps)[-1][0]
            return {f"{last}__sample_weight": sample_weight}

        return {"sample_weight": sample_weight}

    def train(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        overrides: Dict[str, Any] | None = None,
        sample_weight: Optional[np.ndarray] = None,
        notes: str = "",
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        feature_paths: Optional[List[str]] = None,
    ) -> ModelBundle:
        """训练一个候选模型。

        `eval_set` 提供 (X_val, y_val) 时启用**早停** —— 验证集无提升时提前
        停止迭代, 直接削减训练算力 (赛题客观分考察"单位增益消耗的算力")。

        `feature_paths` 传入官方 flatten 的字段路径 —— 随模型持久化, 供
        预测服务做 schema 一致性校验。
        """
        est = self.factory.build(name, overrides)
        t0 = time.perf_counter()
        fit_kwargs: Dict[str, Any] = {}

        # -- 样本权重的路由 -------------------------------------------------
        #
        # sklearn 的 Pipeline **不接受** `sample_weight` 作为 fit 参数
        # (它只认 `stepname__param` 形式)。而回归候选 (mlp / logreg)
        # 都被 `_scaled()` 包成了
        #     TransformedTargetRegressor( Pipeline(StandardScaler, 模型) )
        # 因此直接把 sample_weight 传给 fit 会抛
        #     ValueError: Pipeline.fit does not accept the sample_weight parameter
        #
        # 正确做法: 检测是否为 Pipeline, 若是则把权重路由到**最后一步**
        # (真正的估计器), 否则按原样传。
        #
        # 注意 TransformedTargetRegressor 也要用 `regressor__` 前缀透传,
        # 否则它的内层 Pipeline 收不到权重。
        if sample_weight is not None:
            fit_kwargs.update(self._weight_fit_kwargs(est, sample_weight))

        if eval_set is not None:
            Xv, yv = eval_set
            fit_kwargs["eval_set"] = [(Xv, yv)]
            fit_kwargs["eval_metric"] = "rmse" if self.task == "regression" else "auc"
            # LightGBM 4.x 的早停通过 callbacks 传入, 不是 fit 参数
            rounds = int(self.cfg.get("model.early_stopping_rounds", 30))
            try:
                import lightgbm as lgb

                fit_kwargs["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]
            except Exception:
                fit_kwargs.pop("eval_set", None)
                fit_kwargs.pop("eval_metric", None)
        try:
            est.fit(X, y, **fit_kwargs)
        except (TypeError, ValueError) as exc:
            # 模型族不支持 eval_set / 早停 → 退回朴素训练
            fit_kwargs.pop("eval_set", None)
            fit_kwargs.pop("eval_metric", None)
            fit_kwargs.pop("callbacks", None)
            try:
                est.fit(X, y, **self._weight_fit_kwargs(est, sample_weight))
            except Exception:
                raise exc
        elapsed = time.perf_counter() - t0

        bundle = ModelBundle(
            name=name,
            estimator=est,
            feature_names=list(feature_names),
            task=self.task,
            train_seconds=round(elapsed, 4),
            params=overrides or {},
            train_size=int(len(y)),
            notes=notes,
            feature_paths=list(feature_paths or []),
        )
        return bundle

    def train_with_cv(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_splits: int = 3,
        overrides: Dict[str, Any] | None = None,
    ) -> Tuple[ModelBundle, Dict[str, float]]:
        """带交叉验证的训练 — 用于在候选中挑选, 降低过拟合风险。"""
        if self.task == "regression":
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import StratifiedKFold

            counts = np.bincount(y.astype(int)) if len(y) else np.array([1])
            splitter = StratifiedKFold(
                n_splits=min(n_splits, max(2, int(counts.min()))), shuffle=True, random_state=42
            )
        scores: List[float] = []
        for tr, va in splitter.split(X, y):
            est = self.factory.build(name, overrides)
            est.fit(X[tr], y[tr])
            p = est.predict(X[va])
            scores.append(compute_metrics(y[va], p, self.task)["score"])
        cv_score = float(np.mean(scores)) if scores else 0.0
        bundle = self.train(name, X, y, feature_names, overrides)
        bundle.metrics["cv_score"] = cv_score
        return bundle, {"cv_score": cv_score}


# =====================================================================
# 评估 & 影子对比
# =====================================================================

def compare_bundles(
    a: ModelBundle,
    b: ModelBundle,
    ds: Dataset,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """比较两个候选模型在带标签集上的差异。

    返回 (a_score, b_score, p_value):
      * p_value 来自 McNemar 检验 (两者错误率的配对差异)
      * p 值小说明 b 显著优于 a (或反之), 方向需看分数

    这是"挑战者必须证明自己更好才能替换冠军"的核心判据。
    """
    lab = ds.labeled()
    a_pred = a.predict(lab.X)
    b_pred = b.predict(lab.X)
    a_score = compute_metrics(lab.y, a_pred, a.task)["score"]
    b_score = compute_metrics(lab.y, b_pred, b.task)["score"]
    p = significance_pvalue(a_pred, b_pred, lab.y, a.task)
    return a_score, b_score, p


def evaluate(bundle: ModelBundle, ds: Dataset) -> Dict[str, float]:
    if not ds.has_labels:
        return {}

    lab = ds.labeled()
    pred = bundle.predict(lab.X)
    return compute_metrics(lab.y, pred, bundle.task)


@dataclass
class ShadowResult:
    """影子评估结论 — 决定是否允许上线。"""

    approved: bool
    delta_score: float
    baseline_score: float
    candidate_score: float
    n_samples: int
    p_value: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "approved": self.approved,
            "delta_score": round(self.delta_score, 6),
            "baseline_score": round(self.baseline_score, 6),
            "candidate_score": round(self.candidate_score, 6),
            "n_samples": round(self.n_samples, 2),
            "p_value": round(self.p_value, 6),
            "reason": self.reason,
        }


def shadow_evaluate(
    baseline: ModelBundle,
    candidate: ModelBundle,
    ds: Dataset,
    alpha: float = 0.05,
    min_samples: int = 300,
    margin: float = 0.005,
    metric_fn: Optional[Any] = None,
) -> ShadowResult:
    """在带标签窗口上对比候选与现网模型。

    判定:
      * 样本量达标
      * 候选分数提升超过 margin (受控: 避免噪声级改动上线)
      * 配对检验 (bootstrap 或 McNemar) 显著

    `metric_fn`:
        可选的指标计算函数 `(y_true, y_pred, task) -> Dict[str, float]`,
        用于接入**赛题专用口径** (A2 严重退化漏报 / A3 非对称评分)。
        缺省用通用 `compute_metrics`。

        为什么必须可替换: 若上线判定用通用口径 (acc/f1/auc) 而赛题实际按
        Macro-F1 + 严重退化漏报打分, 就会出现"内部判定通过、实际评分落后"的
        错位 —— 系统优化了错误的指标。
    """
    if not ds.has_labels:
        return ShadowResult(False, 0.0, 0.0, 0.0, len(ds), 1.0, "窗口无标签, 跳过上线判定")

    # 标签延迟场景下只评估标签已到达的样本
    lab = ds.labeled()
    if len(lab) == 0:
        return ShadowResult(False, 0.0, 0.0, 0.0, 0, 1.0, "窗口标签均未到达, 跳过上线判定")

    b_pred = baseline.predict(lab.X)
    c_pred = candidate.predict(lab.X)
    compute = metric_fn or (lambda yt, yp, t: compute_metrics(yt, yp, t))
    b_m = compute(lab.y, b_pred, baseline.task)
    c_m = compute(lab.y, c_pred, candidate.task)
    delta = c_m["score"] - b_m["score"]
    n_eval = len(lab)

    if n_eval < min_samples:
        return ShadowResult(
            False, delta, b_m["score"], c_m["score"], n_eval, 1.0,
            f"样本量不足 ({n_eval} < {min_samples}), 继续观察",
        )
    if delta <= 0:
        return ShadowResult(
            False, delta, b_m["score"], c_m["score"], n_eval, 1.0,
            f"候选未优于现网 (delta={delta:.4f}), 拒绝上线",
        )
    if delta < margin:
        return ShadowResult(
            False, delta, b_m["score"], c_m["score"], n_eval, 1.0,
            f"增益 {delta:.4f} 小于受控门限 {margin}, 不上线 (避免噪声抖动)",
        )

    p_value = significance_pvalue(b_pred, c_pred, lab.y, baseline.task)
    if p_value > alpha:
        return ShadowResult(
            False, delta, b_m["score"], c_m["score"], len(ds), p_value,
            f"增益显著但统计不显著 (p={p_value:.3f} > {alpha}), 拒绝上线",
        )
    return ShadowResult(
        True, delta, b_m["score"], c_m["score"], n_eval, p_value,
        f"增益 {delta:.4f} 且统计显著 (p={p_value:.4f}), 批准上线",
    )


def significance_pvalue(
    a_pred: np.ndarray,
    b_pred: np.ndarray,
    y: np.ndarray,
    task: str = "binary",
) -> float:
    """配对显著性检验 —— 判定 b 是否显著优于 a。

    * 分类: McNemar 检验 (配对二值错误率差异)
    * 回归: Wilcoxon 符号秩检验 (配对误差分布差异)

    返回的 p 值越小, 说明两个模型的差异越显著 (与方向无关,
    方向需结合分数判断)。
    """
    if task == "regression":
        return _wilcoxon_pvalue(a_pred, b_pred, y)
    return _mcnemar_pvalue(a_pred, b_pred, y)


def _wilcoxon_pvalue(a_pred: np.ndarray, b_pred: np.ndarray, y: np.ndarray) -> float:
    """Wilcoxon 符号秩检验 —— 比较两个模型的绝对误差分布。"""
    a_err = np.abs(np.asarray(a_pred, dtype=np.float64) - y)
    b_err = np.abs(np.asarray(b_pred, dtype=np.float64) - y)
    diff = b_err - a_err
    if np.allclose(diff, 0):
        return 1.0
    try:
        from scipy import stats

        # zero_method='wilcox' 丢弃差值为 0 的配对
        return float(stats.wilcoxon(b_err, a_err, zero_method="wilcox").pvalue)
    except Exception:
        # 退化: 用配对 t 检验
        from scipy import stats

        if np.std(diff) < 1e-12:
            return 1.0
        return float(stats.ttest_rel(b_err, a_err).pvalue)


def _mcnemar_pvalue(b_pred: np.ndarray, c_pred: np.ndarray, y: np.ndarray) -> float:
    """McNemar 检验 — 检验两个模型错误率差异是否显著。"""
    b_wrong = b_pred != y
    c_wrong = c_pred != y
    b_only = int(np.sum(b_wrong & ~c_wrong))
    c_only = int(np.sum(~b_wrong & c_wrong))
    n = b_only + c_only
    if n == 0:
        return 1.0
    from scipy import stats

    # 精确二项检验 (双侧)
    return float(stats.binomtest(min(b_only, c_only), n, 0.5).pvalue)


# =====================================================================
# 版本管理 / 回滚
# =====================================================================

class ModelRegistry:
    """模型版本仓库 + 回滚。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.path("paths.artifacts_root") / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self._pointer = self.root / "current.txt"

    def register(self, bundle: ModelBundle, version: str) -> Path:
        bundle.version = version
        path = self.root / f"{version}.pkl"
        bundle.save(path)
        self._pointer.write_text(version, encoding="utf-8")
        return path

    def current_version(self) -> str | None:
        if self._pointer.exists():
            return self._pointer.read_text(encoding="utf-8").strip()
        return None

    def load_current(self) -> ModelBundle | None:
        v = self.current_version()
        if not v:
            return None
        p = self.root / f"{v}.pkl"
        if not p.exists():
            return None
        return ModelBundle.load(p)

    def load(self, version: str) -> ModelBundle | None:
        p = self.root / f"{version}.pkl"
        return ModelBundle.load(p) if p.exists() else None

    def promote(self, version: str) -> None:
        self._pointer.write_text(version, encoding="utf-8")

    def rollback(self, previous_version: str) -> bool:
        """回滚到指定历史版本。"""
        if self.load(previous_version) is None:
            return False
        self.promote(previous_version)
        return True

    def history(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.pkl"))
