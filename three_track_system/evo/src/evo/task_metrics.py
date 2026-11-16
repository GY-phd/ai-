"""赛题专用指标 — 按官方 profile (A/B/C) 精确对齐评分口径。

**权威来源**: 赛题包 `baselines/docs/{A,B,C}.md` 与 `baselines/configs/{A,B,C}.json`。

| profile | 任务 | 主指标 | 辅助指标 | 官方明确不使用 |
|---|---|---|---|---|
| A | 气体 6 类分类 | **Macro-F1** (越高越好) | Accuracy、混淆矩阵 | 不把连续数值当标签 |
| B | 轴承 RUL 回归 (分钟) | **RMSE** (越低越好) | MAE | **不使用严重退化分类/漏报率**、65%/20% 阈值、寿命比例标签 |
| C | 发动机 RUL 回归 (循环) | **RMSE** (越低越好) | MAE、**NASA 非对称分数** | 不做健康/预警/危险分箱、目标不封顶 |

关键设计:
  * **profile B 绝不能用漏报率打分** —— 官方原文"不以严重故障漏报率评价
    本任务"。若把漏报率并入 score, 系统就会去优化一个**不被评分**的量,
    反而损害真正的主指标 RMSE。这是一个"优化了错误目标"的典型陷阱。
  * **profile C 的非对称分数用 NASA 官方公式** (逐样本惩罚之**和**,
    高估惩罚是低估的 5 倍)。官方特别说明"该分数是逐样本惩罚之和,
    样本数量或测试协议不同的结果不能直接横比" → 不能取均值冒充。
  * 严重退化漏报仅作为**可选诊断指标**输出, 不进 score、不否决上线。

所有 score 归一化为 **"越大越好"**, 与 Harness 的影子评估、增益计算、
bandit 奖励保持同一量纲语义。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


# =====================================================================
# A1 / A2: 分类指标
# =====================================================================

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 —— A1 (Accuracy/Macro-F1) 与 A2 (Macro-F1) 的主指标。

    与普通 F1 的区别: Macro 是**逐类 F1 的等权平均**, 不被大类主导。
    工业场景里"严重退化"往往只占 2~5% 的样本, 用普通 F1 (以多数类为分母)
    几乎看不出差别 —— 这正是赛题要 Macro-F1 的原因。
    """
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_true, y_pred))


def severe_degradation_miss_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severe_classes: Optional[Sequence[Any]] = None,
    severe_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """**严重退化漏报率** —— 诊断指标 (官方 B/C 包**不计入评分**)。

    定义:
        漏报 (miss) = 真值属于"严重退化", 但模型**没有把它识别为严重退化**。

    分类场景 (severe_classes 给出):
        预测不属于严重类别 → 漏报。

    回归场景 (severe_threshold 给出, 如"剩余寿命低于 20 分钟即严重退化"):
        严重退化意味着**剩余寿命很短**, 因此"没识别出来" = 预测值
        **高于**严重阈值 (模型以为还很久, 实际快坏了)。
        公式: miss ⟺ y_true ≤ threshold 且 y_pred > threshold

        注意方向: 这里"漏报"指的是**低估了退化严重性**(以为还能用很久),
        与分类场景"把坏的判成好的"语义一致。

    两种场景下都返回:
        {
          "severe_miss_rate": 漏报率 (0~1, 越小越好),
          "severe_miss_count": 漏报个数,
          "severe_count": 严重退化样本总数,
          "severe_recall": 严重退化召回率 (1 - 漏报率),
          "severe_coverage": 严重退化样本占比,
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {
            "severe_miss_rate": 0.0,
            "severe_miss_count": 0,
            "severe_count": 0,
            "severe_recall": 1.0,
            "severe_coverage": 0.0,
        }

    # -- 判定哪些样本属于"严重退化" --
    if severe_classes is not None:
        severe_set = {str(c) for c in severe_classes}
        severe_mask = np.array([str(v) in severe_set for v in y_true])
        pred_is_severe = np.array([str(v) in severe_set for v in y_pred])
    elif severe_threshold is not None:
        thr = float(severe_threshold)
        try:
            t = np.asarray(y_true, dtype=float)
            p = np.asarray(y_pred, dtype=float)
        except (TypeError, ValueError):
            return {
                "severe_miss_rate": 0.0, "severe_miss_count": 0, "severe_count": 0,
                "severe_recall": 1.0, "severe_coverage": 0.0,
            }
        # 真值: 剩余寿命 ≤ 阈值 → 严重退化
        severe_mask = t <= thr
        # 识别为严重退化: 预测也 ≤ 阈值 (即模型也认为快坏了)
        pred_is_severe = p <= thr
    else:
        # 无任何定义 → 用 y_true 的最大值作为"最严重"类别
        # (轴承数据集通常按退化程度编码, 数值越大越接近失效)
        try:
            y_num = np.asarray(y_true, dtype=float)
            severe_mask = y_num >= float(np.nanmax(y_num))
            pred_num = np.asarray(y_pred, dtype=float)
            pred_is_severe = np.nan_to_num(pred_num, nan=-np.inf) >= float(np.nanmax(y_num))
        except (TypeError, ValueError):
            return {
                "severe_miss_rate": 0.0, "severe_miss_count": 0, "severe_count": 0,
                "severe_recall": 1.0, "severe_coverage": 0.0,
            }

    severe_count = int(severe_mask.sum())
    if severe_count == 0:
        # 影子集里没有严重退化样本 → 该指标无信息量, 不能据此否决候选
        return {
            "severe_miss_rate": 0.0,
            "severe_miss_count": 0,
            "severe_count": 0,
            "severe_recall": 1.0,
            "severe_coverage": 0.0,
        }

    miss_mask = severe_mask & ~pred_is_severe
    miss_count = int(miss_mask.sum())
    miss_rate = miss_count / severe_count
    return {
        "severe_miss_rate": float(miss_rate),
        "severe_miss_count": miss_count,
        "severe_count": severe_count,
        "severe_recall": float(1.0 - miss_rate),
        "severe_coverage": float(severe_count / n),
    }


# =====================================================================
# A3: RUL 非对称评分
# =====================================================================

def asymmetric_rul_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    over_predict_penalty: float = 2.0,
    cap: Optional[float] = None,
) -> Dict[str, float]:
    """A3 发动机剩余寿命的**非对称评分**。

    工业语义 (与 NASA C-MAPSS 一致):
        * **过预测寿命** (pred > true): 说"还能用 500 分钟", 实际 100 分钟就断
          → 设备超期服役, 代价是**报废/故障**, 惩罚更重。
        * **欠预测寿命** (pred < true): 说"100 分钟就到寿命了", 实际还能用 500
          → 只是提前换件, 代价是**浪费剩余寿命**。

    惩罚权重:
        * 过预测误差按 `over_predict_penalty` (默认 2.0) 加权。
        * 误差按 `cap` (若给定) 截断 —— 极长寿命的样本误差可达数千分钟,
          不截断会让少数离群点主导整个分数。

    返回:
        {
          "asym_score": 越大越好的综合分 (-加权绝对误差),
          "asym_penalty": 加权绝对误差 (越小越好),
          "over_predict_rate": 过预测样本占比,
          "over_predict_mae": 过预测样本的平均误差,
          "under_predict_mae": 欠预测样本的平均误差,
        }
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {
            "asym_score": 0.0,
            "asym_penalty": 0.0,
            "over_predict_rate": 0.0,
            "over_predict_mae": 0.0,
            "under_predict_mae": 0.0,
        }

    err = y_pred - y_true
    if cap is not None:
        err = np.clip(err, -float(cap), float(cap))
    over = err > 0
    under = ~over

    w = np.where(over, float(over_predict_penalty), 1.0)
    penalty = float(np.mean(np.abs(err) * w))

    out = {
        "asym_penalty": penalty,
        "asym_score": -penalty,
        "over_predict_rate": float(np.mean(over)),
        "over_predict_mae": float(np.mean(np.abs(err[over]))) if over.any() else 0.0,
        "under_predict_mae": float(np.mean(np.abs(err[under]))) if under.any() else 0.0,
    }
    return out


def nasa_asymmetric_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """NASA C-MAPSS 官方非对称误差分数 (任务 C 的辅助指标)。

    官方定义 (逐样本惩罚之**和**, 不是均值!):

        score = Σ_i  {
            (1 / r_i) · max(0, r_i − ŷ_i)      若 ŷ_i < r_i   (低估剩余寿命)
            (5 / r_i) · max(0, ŷ_i − r_i)      否则          (高估剩余寿命)
        }

    语义: **高估剩余寿命的惩罚是低估的 5 倍** —— 说"还能飞 500 循环"
    实际 100 循环就失效, 发动机可能当场报废; 而低估只是提前换发。

    为什么必须用"和"而不是"均值":
        官方明确说"该分数是逐样本惩罚之和, 样本数量或测试协议不同的结果
        不能直接横比"。若取均值, 分数会随样本数变化, 与官方口径不符,
        也无法与排行榜对比。

    返回:
        {
          "nasa_asym_score": 官方分数 (越低越好, 逐样本惩罚之和),
          "nasa_asym_score_mean": 每样本平均惩罚 (便于跨协议比较, 仅供参考),
          "over_predict_rate": 高估样本占比,
          "over_predict_mae": 高估样本的平均误差,
          "under_predict_mae": 低估样本的平均误差,
        }
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {
            "nasa_asym_score": 0.0,
            "nasa_asym_score_mean": 0.0,
            "over_predict_rate": 0.0,
            "over_predict_mae": 0.0,
            "under_predict_mae": 0.0,
        }

    # 分母 r_i: 真实剩余寿命。官方用 max(1, r_i) 防止除零。
    r = np.maximum(np.abs(y_true), 1.0)
    err = y_pred - y_true
    over = err > 0  # 高估剩余寿命 (ŷ > r)

    penalty = np.where(
        over,
        5.0 * np.maximum(0.0, err) / r,
        1.0 * np.maximum(0.0, -err) / r,
    )
    return {
        "nasa_asym_score": float(penalty.sum()),
        "nasa_asym_score_mean": float(penalty.mean()),
        "over_predict_rate": float(np.mean(over)),
        "over_predict_mae": float(np.mean(np.abs(err[over]))) if over.any() else 0.0,
        "under_predict_mae": float(np.mean(np.abs(err[~over]))) if (~over).any() else 0.0,
    }


# =====================================================================
# 统一入口
# =====================================================================

def compute_task_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "binary",
    profile: str = "auto",
    severe_classes: Optional[Sequence[Any]] = None,
    severe_threshold: Optional[float] = None,
    asym_cap: Optional[float] = None,
) -> Dict[str, float]:
    """按赛题官方 profile 计算指标 —— Harness 的统一指标入口。

    **官方 profile 定义** (来自赛题包 baselines/docs/{A,B,C}.md):

    | profile | 任务 | 主指标 | 辅助指标 | 明确不使用 |
    |---|---|---|---|---|
    | A | 气体 6 类分类 | **Macro-F1** (越高越好) | Accuracy、混淆矩阵 | 不把连续数值当标签 |
    | B | 轴承 RUL 回归 | **RMSE** (越低越好) | MAE | **不使用严重退化分类/漏报率** |
    | C | 发动机 RUL 回归 | **RMSE** (越低越好) | MAE、**NASA 非对称分数** | 不做健康/预警/危险分箱 |

    关键约束 (来自官方原文, 踩不得):
      * B: "不以严重故障漏报率评价本任务"、"不使用正常/早期退化/严重退化、
        65%/20% 阈值或寿命比例标签"
      * C: "回归目标不封顶、不做健康/预警/危险等阈值分箱"

    因此本函数**完全不输出**严重退化漏报指标 —— 那是不被评分的量,
    出现在 score 相关输出里会误导"优化了错误目标"。
    如需该诊断, 请单独调用 `severe_degradation_miss_rate`。

    `score` 统一为"越大越好":
      * A (分类): 0.5·Macro-F1 + 0.3·Accuracy + 0.2·AUC
      * B (回归): -RMSE
      * C (回归): -RMSE  (NASA 非对称分数作为辅助指标单独报告)
    """
    from .models import compute_metrics

    base = compute_metrics(y_true, y_pred, task)
    out: Dict[str, float] = dict(base)
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)

    if profile == "auto":
        profile = "C" if task == "regression" else "A"

    if task == "regression":
        # ---- B / C: RUL 回归 ----
        if profile == "C":
            nasa = nasa_asymmetric_score(y_true_a, y_pred_a)
            out.update(nasa)
        else:
            # profile B: 官方明确"不以严重故障漏报率评价本任务"
            # → 只算 RMSE/MAE, score = -RMSE
            if asym_cap is not None:
                # 可选的截断诊断 (不改变 score)
                out["asym_penalty"] = float(
                    np.mean(np.abs(np.clip(y_pred_a - y_true_a, -asym_cap, asym_cap)))
                )
        # 主指标 RMSE (两个 profile 一致)
        out["score"] = -out.get("rmse", 0.0)
        return out

    # ---- A: 分类 (Macro-F1 主指标) ----
    if len(y_true_a) == 0:
        return out
    out["macro_f1"] = macro_f1(y_true_a, y_pred_a)
    out["acc"] = accuracy(y_true_a, y_pred_a)
    # 混淆矩阵 (官方辅助指标)
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true_a, y_pred_a, labels=sorted(set(y_true_a.tolist())))
        out["confusion_matrix"] = cm.tolist()
    except Exception:
        pass
    # score 严格按官方口径: 主指标 Macro-F1 + 辅助 Accuracy。
    #
    # **不掺入 AUC**: 官方 A 包明确"主指标为 Macro-F1, 辅助查看 Accuracy
    # 和混淆矩阵", 没有 AUC 这一项。而多分类下若只有硬标签, ROC-AUC 会
    # 退化到 0.5 —— 掺进去等于给 score 加了一个恒定的负偏置, 却不改变
    # 模型的相对优劣 (所有候选一起被扣同样的分), 只会让 margin 门限
    # 的绝对值失去意义。
    out["score"] = 0.7 * out["macro_f1"] + 0.3 * out["acc"]
    return out
