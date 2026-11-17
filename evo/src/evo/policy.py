"""策略层自改进 (策略层自进化).

核心思想:
  更新决策本身也是需要被优化的对象。用一个受限离散 bandit (Thompson 采样)
  在"更新决策参数空间"上做在线选择, 用"单位增益消耗的算力/标注/Token"作为成本,
  以"性能增益 / 成本"作为奖励 —— 这正是赛题客观分的核心度量。

受控机制 (Safety):
  * 参数只能在离散候选空间内移动, 单步位移受限 (max_param_shift)
  * 策略更新需经过"影子评估"才允许生效
  * 预算硬上限 + 连续失败熔断 + 性能回退自动回滚
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config


# =====================================================================
# 策略参数
# =====================================================================

@dataclass
class PolicyState:
    """当前生效的策略参数。

    分为两组, 职责不同:
      * 决策参数 (decision): psi_trigger / min_gain —— 决定**什么时候**更新、
        多大的增益才值得一次更新。属于"策略层", 由 champion 统一持有。
      * 模型参数 (model): retrain_window / lr / num_leaves —— 决定**怎么**训练。
        属于"模型层", 每次更新都可以由 bandit 提名多个候选, 在影子空间里比选。
    """

    # -- 决策参数 (champion 级) --
    psi_trigger: float = 0.1
    min_gain: float = 0.005
    # -- 模型参数 (可被 bandit 探索) --
    retrain_window: int = 2000
    lr: float = 0.05
    num_leaves: int = 31

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psi_trigger": self.psi_trigger,
            "min_gain": self.min_gain,
            "retrain_window": self.retrain_window,
            "lr": self.lr,
            "num_leaves": self.num_leaves,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PolicyState":
        base = PolicyState()
        for k, v in (d or {}).items():
            if hasattr(base, k) and v is not None:
                setattr(base, k, v)
        return base

    @staticmethod
    def decision_keys() -> List[str]:
        return ["psi_trigger", "min_gain"]

    @staticmethod
    def model_keys() -> List[str]:
        return ["retrain_window", "lr", "num_leaves"]


@dataclass
class Action:
    """一次更新动作及其成本收益记录。"""

    arm_index: int
    policy: PolicyState
    gain: float = 0.0
    cost_gpu_seconds: float = 0.0
    cost_labels: int = 0
    cost_tokens: int = 0
    approved: bool = False
    note: str = ""
    # 是否真正尝试了一次更新 (monitor/跳过 不算尝试, 不计入熔断与 bandit)
    attempted: bool = True

    @property
    def total_cost(self) -> float:
        """归一化总成本。GPU 秒为基准, 标注与 Token 按经验权重折算。"""
        return (
            self.cost_gpu_seconds
            + 0.02 * self.cost_labels
            + 0.0005 * self.cost_tokens
        )

    @property
    def efficiency(self) -> float:
        """单位成本增益 — 对齐赛题客观分 '单位增益消耗的算力'。"""
        if self.total_cost <= 1e-9:
            return 0.0
        return self.gain / self.total_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_index": self.arm_index,
            "policy": self.policy.to_dict(),
            "gain": round(self.gain, 6),
            "cost_gpu_seconds": round(self.cost_gpu_seconds, 4),
            "cost_labels": self.cost_labels,
            "cost_tokens": self.cost_tokens,
            "total_cost": round(self.total_cost, 6),
            "efficiency": round(self.efficiency, 8),
            "approved": self.approved,
            "note": self.note,
        }


# =====================================================================
# 受限 Thompson 采样 bandit
# =====================================================================

class ParameterBandit:
    """单参数受限 Thompson 采样。

    设计取舍 (重要):
      最初把 5 个参数做笛卡尔积产生 720 个臂, 在几十轮迭代内**无法完成有意义的
      探索** —— 实测表现为 bandit 一直停在起始臂, 策略层退化为常量,
      反而不如固定超参 (消融实验 E2 vs E3 直接暴露了这个问题)。

      因此改为 **每参数一个独立 bandit**: 5 个 bandit, 每个 3~5 个臂,
      总共 19 个动作, 在 30 轮内可充分探索, 且天然满足"单步位移受限"
      (每个参数每次只允许移动到相邻档位, 即受控 RSI 的约束)。
    """

    def __init__(self, name: str, values: List[Any], seed: int = 0):
        self.name = name
        self.values = list(values)
        self.rng = random.Random(seed + hash(name) % 10000)
        self.index = 0
        # 每个臂: [奖励样本列表]
        self.stats: List[List[float]] = [[] for _ in self.values]

    @property
    def value(self) -> Any:
        return self.values[self.index]

    def select(self) -> int:
        """Thompson 采样, **仅允许在相邻档位间移动** (受控)。

        选中即采纳 (index 立即前进), 保证外部拿到的 PolicyState
        与实际生效的档位一致; 奖励在 update() 中回灌到该档位。
        """
        lo = max(0, self.index - 1)
        hi = min(len(self.values) - 1, self.index + 1)
        cand = list(range(lo, hi + 1))
        best_i, best_p = cand[0], -1.0
        for i in cand:
            rewards = self.stats[i]
            if rewards:
                mean = sum(rewards) / len(rewards)
                n = len(rewards)
                # Beta 后验: 均值映射到 [0,1], 样本量决定置信度
                a = 1.0 + mean * n
                b = 1.0 + max(0.0, 1.0 - mean) * n
                p = self.rng.betavariate(max(a, 0.01), max(b, 0.01))
            else:
                # 未探索臂: 给一个随机的乐观先验, 保证会去试
                p = 1.0 + self.rng.random()
            if p > best_p:
                best_p, best_i = p, i
        self.index = best_i
        return best_i

    def update(self, index: int, reward: float) -> None:
        if 0 <= index < len(self.stats):
            self.stats[index].append(float(np.clip(reward, 0.0, 1.0)))
            self.index = index

    def summary(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "index": self.index,
            "trials": [len(s) for s in self.stats],
            "means": [
                (round(sum(s) / len(s), 6) if s else None) for s in self.stats
            ],
        }


class RestrictedThompsonBandit:
    """参数化策略搜索器 — 一组独立单参数 bandit。

    每次 select() 在每个参数上各做一次受限 Thompson 采样,
    组合成一个完整的 PolicyState。这使 30 轮内可以同时探索所有参数维度。
    """

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.space: Dict[str, List[Any]] = cfg.get("policy.param_space", {}) or {}
        self.exploration = float(cfg.get("policy.exploration_rate", 0.35))
        self.max_shift = int(cfg.get("policy.max_param_shift", 1))

        self.params: Dict[str, ParameterBandit] = {
            k: ParameterBandit(k, v, seed=seed) for k, v in self.space.items()
        }
        # 初始策略对齐规则:
        #   1) 优先对齐到 PolicyState 的默认值所在档位 (默认值在空间内时)
        #   2) 默认值不在空间内 → 对齐到第一档
        # 这样 `cfg.set("policy.param_space.psi_trigger", [0.0001])` 这类覆盖
        # 能真正生效 (默认值 0.1 不在空间内 → 落到 0.0001), 而不会静默被忽略。
        default = PolicyState()
        for k, pb in self.params.items():
            dv = getattr(default, k, None)
            if dv in pb.values:
                pb.index = pb.values.index(dv)
            else:
                pb.index = 0

        self.history: List[Action] = []
        self._last_selection: Dict[str, int] = {k: pb.index for k, pb in self.params.items()}

    # -- 兼容旧接口 ------------------------------------------------------
    @property
    def arms(self) -> List[PolicyState]:
        """当前可枚举的策略组合 (仅用于展示, 不做精确搜索)。"""
        return [PolicyState(**{k: pb.value for k, pb in self.params.items()})]

    @property
    def current_index(self) -> int:
        return sum(pb.index for pb in self.params.values())

    def _distance(self, a: PolicyState, b: PolicyState) -> int:
        d = 0
        for key, pb in self.params.items():
            va, vb = getattr(a, key), getattr(b, key)
            if va in pb.values and vb in pb.values:
                d = max(d, abs(pb.values.index(va) - pb.values.index(vb)))
        return d

    # -- 选择 ------------------------------------------------------------
    def select(self) -> Tuple[int, PolicyState]:
        """每个参数独立做一次受限 Thompson 采样, 组合成完整策略。"""
        chosen: Dict[str, Any] = {}
        for name, pb in self.params.items():
            i = pb.select()
            self._last_selection[name] = i
            chosen[name] = pb.values[i]
        state = PolicyState(**chosen)
        return self.current_index, state

    # -- 更新 ------------------------------------------------------------
    def feedback(
        self,
        variant: "PolicyState",
        arm_index: int,
        reward: float,
        baseline: Optional["PolicyState"] = None,
    ) -> List[str]:
        """按**变体实际改了哪个维度**回灌奖励。

        每个变体只改动一个参数维度, 因此奖励可以精确归因到该维度,
        bandit 才能学到 "lr=0.02 更好" 这类可解释的结论。

        关键: **未改动的维度绝不能收到奖励**。
        若把变体的奖励同时记到 champion 在其他维度上的取值上,
        会污染这些臂的后验统计 —— bandit 会误以为"lr=0.02 且 leaves=31 很好",
        而真实因果只是 lr 变了。这会让策略层的学习结果不可解释。

        返回: 实际收到奖励的维度名列表 (便于审计与调试)。
        """
        r = float(np.clip(reward, 0.0, 1.0))
        credited: List[str] = []
        for k, pb in self.params.items():
            v = getattr(variant, k, None)
            if v not in pb.values:
                continue
            if baseline is not None:
                base_v = getattr(baseline, k, None)
                if base_v is not None and base_v == v:
                    # 该维度相对基线没有变化 —— 不归因
                    continue
            pb.update(pb.values.index(v), r)
            credited.append(k)
        return credited

    def update(self, action: Action) -> None:
        """把这次执行的收益回灌到**所有被选中的参数臂**上。"""
        reward = self._normalize(action)
        for name, pb in self.params.items():
            i = self._last_selection.get(name, pb.index)
            pb.update(i, reward)
        self.history.append(action)

    @staticmethod
    def _normalize(action: Action) -> float:
        """把 efficiency 归一化到 [0,1] 作为 bandit 奖励。

        绝对 efficiency 量级很小 (~1e-3), 直接用会把所有臂都压到 0,
        导致 bandit 无法区分优劣。因此按"单位成本增益"的相对大小做压缩映射。
        """
        eff = max(0.0, action.efficiency)
        # 1e-3 附近的 efficiency 映射到 ~0.5, 便于 Beta 后验区分
        return float(1.0 - math.exp(-eff / 2e-3))

    def summary(self) -> Dict[str, Any]:
        return {
            "params": {k: pb.summary() for k, pb in self.params.items()},
            "n_actions": len(self.history),
        }

    def set_state(self, state: PolicyState) -> None:
        """把各参数档位强制对齐到给定策略 (用于冠军-挑战者机制的回退)。"""
        for k, pb in self.params.items():
            v = getattr(state, k, None)
            if v in pb.values:
                pb.index = pb.values.index(v)
                self._last_selection[k] = pb.index


# =====================================================================
# 策略控制器
# =====================================================================

@dataclass
class BudgetState:
    """预算使用状态 — 受控 RSI 的硬约束。"""

    gpu_seconds: float = 0.0
    labels: int = 0
    tokens: int = 0
    limits: Dict[str, float] = field(default_factory=dict)

    def usage_ratio(self) -> float:
        if not self.limits:
            return 0.0
        parts = []
        for key, used in (
            ("gpu_seconds", self.gpu_seconds),
            ("labels", self.labels),
            ("tokens", self.tokens),
        ):
            lim = self.limits.get(key, 0.0)
            if lim:
                parts.append(used / lim)
        return max(parts) if parts else 0.0

    def labels_remaining(self) -> float:
        return max(0.0, float(self.limits.get("labels", 1e18)) - self.labels)

    def exhausted(self) -> bool:
        """是否必须停止主动更新。

        注意: 标注配额耗尽**不等于**系统停摆 —— 已解出的历史标注仍可复用
        (重放缓冲), 此时只是不能再"购买"新标注。真正让系统停摆的是
        算力或 Token 耗尽。
        """
        if not self.limits:
            return False
        halt = float(self.limits.get("halt_ratio", 0.9))
        gpu_lim = self.limits.get("gpu_seconds", 0.0)
        tok_lim = self.limits.get("tokens", 0.0)
        if gpu_lim and self.gpu_seconds / gpu_lim >= halt:
            return True
        if tok_lim and self.tokens / tok_lim >= 1.0:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_seconds": round(self.gpu_seconds, 3),
            "labels": self.labels,
            "tokens": self.tokens,
            "usage_ratio": round(self.usage_ratio(), 4),
        }


class PolicyController:
    """策略层控制器。

    职责:
      1. 根据漂移报告判断是否需要更新 (决策)
      2. 选择候选模型与训练超参 (模型层动作)
      3. 记录成本收益, 用 bandit 反馈优化自身的决策参数 (策略层自改进)

    **冠军-挑战者机制 (Champion/Challenger)** —— 这是策略层"受控"的关键:

      实测发现, 若把 bandit 探索出的策略**直接**用于线上模型, 探索期会
      造成明显的性能抖动 (对照实验 E3 反而不如固定超参的 E2)。
      正确做法是:
        * 维护一个 champion 策略 (当前生效, 已证明有效)
        * bandit 每次提出一个 challenger 策略
        * challenger 只有在其引导的训练 + 影子评估**通过**后, 才晋升为 champion
        * 未通过则 champion 保持不变, 只把失败作为负反馈喂给 bandit
      这样策略层的探索永远不会污染线上性能, 符合受控 RSI 的要求。
    """

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.bandit = RestrictedThompsonBandit(cfg, seed=seed)
        # champion: 当前生效策略; challenger: 待验证的探索策略
        #
        # 初始策略必须**对齐到配置里参数空间的起始档**, 而不是硬编码默认值。
        # 否则 `cfg.set("policy.param_space.psi_trigger", [0.0001])` 这类覆盖
        # 完全不生效 —— 初始 state 仍是默认的 0.1, 导致对照实验里的
        # "定时重训"变体实际退化成了"漂移触发重训"(与完整系统行为雷同)。
        self.state = PolicyState()
        self.champion = PolicyState()
        for key in PolicyState.decision_keys() + PolicyState.model_keys():
            values = (cfg.get("policy.param_space", {}) or {}).get(key)
            if not values:
                continue
            default = getattr(PolicyState(), key)
            # 对齐到与默认值同名的档位; 没有同名档则取第一档
            target = default if default in values else values[0]
            setattr(self.state, key, target)
            setattr(self.champion, key, target)
        self.champion_efficiency = 0.0
        self.champion_gain = 0.0
        self.bandit.set_state(self.champion)
        self._exploring = False
        self._champion_locked = False
        # 探索维度的轮转游标 (保证每个模型参数维度都有机会被探索)
        self._explore_cursor = 0
        self.audit_note = ""
        # 探索轮次的随机源 (与 bandit 各自的 rng 独立, 保证门控可复现)
        self._rng = random.Random(seed + 12345)
        # ---- 探索预算自适应 ----
        #
        # 依据: RUL 赛题的对照实验实测, 探索率从 0 提到 0.5 时,
        # 尾段 RMSE 只从 194.8 改善到 193.9 分钟 (0.5%), 而 GPU 成本
        # 从 13.8s 涨到 46.7s (238%)。**探索的期望收益不抵算力成本**。
        #
        # 但这不能靠"关掉探索"解决 —— 那样就退化成固定超参, 失去了
        # 自进化的意义。正确做法是让探索预算**自我调节**:
        #   * 探索带来了改进 → 升频 (值得继续投)
        #   * 探索没有改进   → 降频 (把算力还给主线)
        # 这样探索的期望收益自动收敛到成本线附近, 形成闭环的成本反馈。
        self._explore_rate = float(cfg.get("policy.exploration_rate", 0.25))
        self._explore_rate_min = float(cfg.get("policy.exploration_rate_min", 0.05))
        self._explore_rate_max = float(cfg.get("policy.exploration_rate_max", 0.5))
        self._explore_gain_ema = 0.0  # 探索带来的增益 (指数滑动平均)
        self._n_explore_trials = 0
        self._n_explore_wins = 0
        self.budget = BudgetState(
            limits={
                "gpu_seconds": float(cfg.get("budget.max_gpu_seconds", 3600)),
                "labels": float(cfg.get("budget.max_label_quota", 2000)),
                "tokens": float(cfg.get("budget.max_llm_tokens", 200000)),
                "halt_ratio": float(cfg.get("budget.halt_ratio", 0.9)),
            }
        )
        self.max_consecutive_failures = int(cfg.get("safety.max_consecutive_failures", 3))
        self.consecutive_failures = 0
        self.halted = False
        # 允许完全关闭主动更新 (用于消融: 冻结静态基线)
        self.updates_enabled = bool(cfg.get("policy.updates_enabled", True))
        self.alarm_triggers = bool(cfg.get("policy.alarm_always_triggers", True))
        # 定时重训模式 (用于消融 E1): 无视漂移强度, 每轮都更新。
        # 与"把 psi_trigger 设成极小值"**不等价** —— 极小阈值仍受
        # 熔断/预算约束, 且一旦 psi_trigger 被策略层调整就会失效。
        # 这里用独立开关保证 E1 确实是"固定周期重训"的对照。
        self.periodic_retrain = bool(cfg.get("policy.periodic_retrain", False))
        # -- 决策层策略学习 --
        # 记录"每个 psi_trigger 档位下, 触发更新的平均收益",
        # 使"何时更新"这一决策本身也能被优化 (而不是永远固定 0.1)。
        self._trigger_stats: Dict[float, List[float]] = {}
        # 各档位的**反事实触发率** (纯观测, 不改变线上行为)
        self._trigger_rate: Dict[float, List[float]] = {}
        self._last_action_outcome: Optional[Tuple[float, float]] = None
        # 当前生效的探针阈值 (None 表示用 champion 阈值)
        self._probe_threshold: Optional[float] = None
        # 距上次探针已过的步数 (控制探针频率)
        self._steps_since_probe = 0
        # 探针轮转游标 (保证 champion 档位与探针档位交替出场)
        self._probe_turn = 0
        # 决策参数最小调整间隔 (步) —— 防止阈值抖动
        self.decision_update_interval = int(cfg.get("policy.update_interval", 200))
        self._steps_since_decision_update = 0
        # 决策层探索的最小收益差 (超过才切换 champion)
        self.decision_min_improvement = float(
            cfg.get("policy.decision_min_improvement", 0.005)
        )

    # -- 决策 ------------------------------------------------------------
    def probe_trigger_threshold(self) -> Optional[float]:
        """**探针**: 轮流在不同的 psi_trigger 档位上取样, 让决策层有数据可学。

        为什么需要探针:
          如果永远只用当前 champion 的阈值 (0.1) 触发更新, 那么其余档位
          (0.08 / 0.15 / 0.2 / 0.25) 的 bandit 样本数恒为 0 ——
          "何时更新"这个决策从未被比较过, 谈不上自改进。
          实测产物里 `psi_trigger` 的 trials 全为 0, 正是这个原因。

        探针的安全约束:
          * 只在**没有正在进行的更新**时取样 (不会打断在途更新);
          * 只切换阈值, 不切换模型超参 (模型超参由 champion 保护);
          * 探针期间仍受影子评估与熔断保护 —— 探索不会污染线上性能。
        """
        pb = self.bandit.params.get("psi_trigger")
        if pb is None or len(pb.values) < 2:
            return None
        if self._champion_locked:
            return None
        # 探针频率受控: 两次探针之间至少间隔 `decision_probe_interval` 步,
        # 否则阈值会每轮抖动, 更新频率忽高忽低, 线上性能不稳。
        interval = int(self.cfg.get("policy.decision_probe_interval", 3))
        if self._steps_since_probe < interval:
            return None
        # 受控: 只在相邻档位间移动
        cur = float(self.state.psi_trigger)
        if cur not in pb.values:
            return None
        i = pb.values.index(cur)
        lo, hi = max(0, i - 1), min(len(pb.values) - 1, i + 1)
        choices = [j for j in range(lo, hi + 1) if j != i]
        if not choices:
            return None
        j = self._rng.choice(choices)
        self._probe_threshold = float(pb.values[j])
        self._steps_since_probe = 0
        return self._probe_threshold

    def probe_or_keep(self) -> Optional[float]:
        """决定本轮是否探针。

        **探针是纯观测, 不改变任何行为**。

        早期版本让探针直接替换当轮的触发阈值, 结果线上行为被扰动:
        探针轮用更低阈值 (0.08) 触发、champion 轮用 0.1 触发, 两者的
        更新时机错开, champion 参数得不到连续验证 —— 实测 tail_score
        从 0.5446 掉到 0.5088, 纯属探索扰动的代价。

        正确做法: 探针**只在"本来就会触发更新"的轮次上**记录一次
        对照观测 —— 即 PSI 已经超过 champion 阈值之后, 再用探针档位
        独立判断"若阈值是 X 会不会也触发", 把这个反事实结果记进决策层。
        这样:
          * 线上行为完全不变 (零扰动);
          * 决策层仍然能比较不同档位的触发频率与收益;
          * 探索成本降为 0。
        """
        if self._champion_locked:
            return None
        self._probe_turn += 1
        return None

    def counterfactual_trigger(self, psi_max: float, severity: str) -> Dict[str, float]:
        """计算"若阈值是各档位, 本轮会不会触发"的反事实对照。

        返回 {阈值: 1.0/0.0} —— 1.0 表示该阈值会触发更新。
        这些反事实观测让决策层能比较不同阈值的**触发频率**,
        而不需要真的改变线上行为。
        """
        out: Dict[str, float] = {}
        pb = self.bandit.params.get("psi_trigger")
        if pb is None:
            return out
        for t in pb.values:
            t = float(t)
            fires = 1.0 if (psi_max >= t or (severity == "alarm" and self.alarm_triggers)) else 0.0
            out[t] = fires
        return out

    def observe_counterfactual(self, psi_max: float, severity: str) -> None:
        """把反事实触发结果记入决策层统计 (不改变线上行为)。"""
        if self.halted:
            return
        cf = self.counterfactual_trigger(psi_max, severity)
        for t, fires in cf.items():
            # 只在"会触发"的档位上留正样本, "不会触发"的留 0 样本 ——
            # 这样每档的均值 = 该阈值下的触发率, 可直接用于比较。
            self._trigger_rate.setdefault(t, []).append(fires)

    def end_probe(self) -> None:
        """结束本轮的探针, 恢复 champion 阈值。"""
        self._probe_threshold = None
        self._steps_since_probe += 1

    def observe_trigger_outcome(
        self, psi_max: float, severity: str, reward: float
    ) -> None:
        """把一次更新的收益归因到**当前生效的阈值档位**。

        决策层的因果链是:
            "用阈值 T 触发了一次更新" → "这次更新带来了收益 R"
        因此奖励记到**当前生效的 T** 上。

        这里只记录"实际发生了更新"的档位 (即当前 champion 阈值);
        漏检的负样本由 `observe_missed_outcome` 补齐。
        """
        if self.halted:
            return
        t = float(self.state.psi_trigger)
        r = float(np.clip(reward, 0.0, 1.0))
        self._trigger_stats.setdefault(t, []).append(r)
        # 同时回灌到 bandit 的 psi_trigger 臂 —— 让该维度的后验统计可用
        pb = self.bandit.params.get("psi_trigger")
        if pb is not None and t in pb.values:
            pb.update(pb.values.index(t), r)

    def observe_missed_outcome(self, psi_max: float) -> None:
        """记录一次"漏检": 漂移已发生但当前阈值没有触发更新。

        这是决策层的**负样本**。若只记录成功更新, 高阈值档位因为漏检
        而样本稀少, 永远拿不到"它其实很糟"的证据 —— 决策层会一直
        偏向高阈值 (漏检是零成本, 但代价是现网模型带着退化裸奔)。
        """
        if self.halted:
            return
        t = (
            float(self._probe_threshold)
            if self._probe_threshold is not None
            else float(self.state.psi_trigger)
        )
        # 漏检的收益记 0 (未触发更新的档位同样要留下证据)
        self._trigger_stats.setdefault(t, []).append(0.0)

    def tick_decision(self) -> None:
        """推进决策层的冷却计数 (每轮调用一次, 无论是否发生更新)。

        冷却期必须按**实际经过的轮数**计, 而不是"发生更新的轮数" ——
        否则在漂移稀疏的场景下冷却期永远走不完, 决策层无法切换阈值。
        """
        self._steps_since_decision_update += 1

    def update_decision_policy(self) -> Optional[str]:
        """根据历史收益, 调整决策参数 (psi_trigger)。

        这是策略层"自改进决策"的核心 —— 之前 `psi_trigger` / `min_gain`
        的 bandit 臂样本数恒为 0 (整个 `plan_action` 从未被调用),
        意味着系统永远用固定阈值 0.1 触发, "何时更新"从未被优化过。

        返回: 变更说明 (无变更则 None)。
        """
        if not self._trigger_stats:
            return None
        # 只在有足够间隔时才允许调整 (防止阈值抖动)
        if self._steps_since_decision_update < self.decision_update_interval:
            return None

        cur_t = float(self.state.psi_trigger)

        # ---- 判据: **累计收益** (而非均值) ----
        #
        # 为什么用累计收益: 低阈值会多触发很多次, 每次收益可能略低,
        # 但**总量**可能更高; 高阈值漏掉大量漂移, 累计收益就低。
        # 用均值会让"触发次数多"看起来像"平均质量差", 判定反向。
        #
        # 同时用**反事实触发率**做交叉校验: 若某档位的触发率与 champion
        # 几乎一致, 那么两者行为等价, 切换没有意义 (保持现状更稳)。
        totals = {t: sum(v) for t, v in self._trigger_stats.items() if v}
        if not totals:
            return None
        best_t = max(totals, key=lambda k: totals[k])

        # champion 档位必须自己有样本, 否则无法比较
        if cur_t not in totals:
            self._steps_since_decision_update = 0
            return None

        # 最小样本量: 样本太少时差异多半是噪声
        min_n = int(self.cfg.get("policy.decision_min_samples", 4))
        if len(self._trigger_stats.get(best_t, [])) < min_n:
            self._steps_since_decision_update = 0
            return None
        if len(self._trigger_stats.get(cur_t, [])) < min_n:
            self._steps_since_decision_update = 0
            return None

        # 累计收益差不足 → 保持现状 (避免噪声驱动的抖动)
        if totals[best_t] - totals[cur_t] < self.decision_min_improvement:
            self._steps_since_decision_update = 0
            return None

        # 触发率几乎相同 → 两者行为等价, 切换只会带来扰动, 不予切换
        rates = self._trigger_rate
        if cur_t in rates and best_t in rates:
            r_cur = sum(rates[cur_t]) / max(1, len(rates[cur_t]))
            r_best = sum(rates[best_t]) / max(1, len(rates[best_t]))
            if abs(r_cur - r_best) < 1e-6:
                self._steps_since_decision_update = 0
                return None

        pb = self.bandit.params.get("psi_trigger")
        if pb is None or best_t not in pb.values:
            self._steps_since_decision_update = 0
            return None

        old_t = cur_t
        self.state.psi_trigger = best_t
        self.champion.psi_trigger = best_t
        if pb.values:
            pb.index = pb.values.index(best_t)
            self.bandit._last_selection["psi_trigger"] = pb.index
        self._steps_since_decision_update = 0
        return (
            f"决策层自适应: psi_trigger {old_t} → {best_t} "
            f"(累计收益 {totals[best_t]:.6f} vs {totals.get(cur_t, 0.0):.6f})"
        )

    def should_update(
        self, psi_max: float, severity: str, force: bool = False
    ) -> Tuple[bool, str]:
        """是否触发一次更新。

        `force=True` 用于**门槛保障探索** (赛题: 至少三次尝试迭代):
        跳过"PSI 是否达阈值"的判断, 但预算、熔断、周期性开关等**硬约束
        仍然全部生效** —— 探索不能绕过安全边界。
        """
        if self.budget.exhausted():
            return False, f"预算使用率 {self.budget.usage_ratio():.1%} 达上限, 仅监控"
        if not self.updates_enabled:
            return False, "主动更新已被配置禁用 (纯监控模式)"
        if self.periodic_retrain:
            return True, "定时重训模式: 无视漂移强度, 按固定周期更新"
        # 当前生效的触发阈值 (可能是决策层的探针档位)
        trigger = (
            self._probe_threshold
            if self._probe_threshold is not None
            else float(self.state.psi_trigger)
        )
        if self.halted:
            # 熔断后的**分级处置** (关键):
            #
            # 熔断的语义是"停止在无效方向上继续探索", 而**不是**"停止一切更新"。
            # 若两者等同, 一旦连续失败触发熔断, 系统在后续窗口全部空转 ——
            # 实测在漂移持续恶化 (PSI 达 alarm) 时也完全不再重训, 现网模型
            # 只能带着退化一路裸奔到流结束, 尾段得分毫无改善。
            #
            # 正确处置:
            #   * severity == alarm (漂移严重, 现网模型确实在退化)
            #     → 允许用**已验证的 champion 参数**继续重训 (低成本、无探索风险),
            #       这是"止损"而非"探索"。
            #   * 其余情况 → 保持只监控, 等待漂移自然消退。
            #
            # 门槛探索在熔断期间**不豁免** —— 熔断意味着连续失败, 此时强行重训
            # 既无意义又违反"熔断后停止在无效方向探索"的语义。
            if severity == "alarm" and psi_max >= trigger:
                return True, (
                    f"熔断后止损模式: PSI {psi_max:.3f} 达 alarm, "
                    f"沿用已验证的 champion 参数重训 (不做新参数探索)"
                )
            return False, "已熔断, 停止主动更新"
        if force:
            return True, (
                f"门槛保障探索: 绕过 PSI 阈值 ({psi_max:.3f} < {trigger}), "
                "但仍受预算/熔断约束"
            )
        if psi_max >= trigger:
            return True, f"PSI {psi_max:.3f} ≥ 触发阈值 {trigger}"
        if severity == "alarm" and self.alarm_triggers:
            return True, f"漂移严重度=alarm (PSI {psi_max:.3f})"
        return False, f"PSI {psi_max:.3f} < 触发阈值 {trigger}, 继续监控"

    def propose_model_candidates(
        self, n: int = 3
    ) -> List[Tuple[int, PolicyState]]:
        """提名候选模型超参 (模型层自迭代的探索)。

        设计要点 —— **候选数随探索预算自适应**:
          对照实验实测: 每轮固定训练 3 个候选会让 GPU 成本变成固定超参方案的
          3 倍, 而收益不足以覆盖成本 (完整系统 E3 的 tail_score 追平但
          efficiency 明显低于 E2)。
          因此:
            * 非探索轮 → 只训练 champion (1 个模型, 成本 = 基线)
            * 探索轮   → champion + (n-1) 个变体 (成本 = n 倍, 但换来学习信号)

        同时保证 **champion 始终在候选集中** —— 挑战者必须证明自己更好
        才能替换它, 探索不会污染线上性能。

        返回: [(arm_index, PolicyState), ...]
        """
        cands: List[Tuple[int, PolicyState]] = []

        # 1) champion 永远在候选集里 (保证有稳定下限)
        champ_state = PolicyState(
            psi_trigger=self.state.psi_trigger,
            min_gain=self.state.min_gain,
            retrain_window=self.champion.retrain_window,
            lr=self.champion.lr,
            num_leaves=self.champion.num_leaves,
        )
        cands.append((self._champion_arm_index, champ_state))

        # 2) 非探索轮不额外提名变体 (成本控制 —— 对齐赛题"单位增益消耗的算力")
        if self._champion_locked:
            return cands
        rate = self.current_exploration_rate()
        if self._rng.random() >= rate:
            return cands

        # 3) 探索轮: 提名 n-1 个变体, 每个只扰动一个模型参数维度 (可归因)
        explored = 0
        attempts = 0
        while explored < max(0, n - 1) and attempts < max(4, n * 3):
            attempts += 1
            variant = self._propose_variant(champ_state)
            if variant is None:
                continue
            arm = self._variant_arm_index(variant)
            if any(a == arm for a, _ in cands):
                continue
            cands.append((arm, variant))
            explored += 1
        return cands

    def _propose_variant(self, base: PolicyState) -> PolicyState | None:
        """构造一个只改动单个模型参数维度的变体 (可归因)。

        维度选择采用**轮转 (round-robin)** 而非随机 shuffle:
          随机 shuffle 会反复命中同一个维度 (实测 8 次探索全部落在 `lr`,
          `retrain_window` 从未被探索) —— bandit 某些臂的样本数恒为 0,
          后验退化成先验, 该维度等于没在学。
          轮转保证每个模型维度都有机会被探索, 且探索预算被均匀分配。
        """
        keys = PolicyState.model_keys()
        if not keys:
            return None
        # 轮转游标: 每提一次名就前进一格
        start = self._explore_cursor % len(keys)
        for offset in range(len(keys)):
            k = keys[(start + offset) % len(keys)]
            pb = self.bandit.params.get(k)
            if pb is None or len(pb.values) < 2:
                continue
            cur = getattr(base, k)
            if cur not in pb.values:
                continue
            i = pb.values.index(cur)
            # 邻域内随机取一档 (受控: 不跳档)
            lo = max(0, i - 1)
            hi = min(len(pb.values) - 1, i + 1)
            choices = [j for j in range(lo, hi + 1) if j != i]
            if not choices:
                continue
            j = self._rng.choice(choices)
            variant = PolicyState(
                psi_trigger=base.psi_trigger,
                min_gain=base.min_gain,
                retrain_window=base.retrain_window,
                lr=base.lr,
                num_leaves=base.num_leaves,
            )
            setattr(variant, k, pb.values[j])
            self._explore_cursor += 1
            return variant
        self._explore_cursor += 1
        return None

    def _variant_arm_index(self, state: PolicyState) -> int:
        idx = 0
        for k, pb in self.bandit.params.items():
            v = getattr(state, k)
            if v in pb.values:
                idx += pb.values.index(v)
        return idx

    def current_exploration_rate(self) -> float:
        """当前的探索预算比例 (自适应)。"""
        return float(self._explore_rate)

    def observe_explore_outcome(self, improved: bool, gain: float) -> None:
        """根据探索轮的结果调节探索预算 (成本-收益闭环反馈)。

        * 探索带来改进 → 提高探索频率 (探索有价值, 值得多投)
        * 探索无改进   → 降低探索频率 (算力不划算, 还给主线)

        采用指数滑动平均评估探索收益, 避免单轮噪声导致预算剧烈抖动。
        """
        self._n_explore_trials += 1
        if improved:
            self._n_explore_wins += 1
        self._explore_gain_ema = 0.8 * self._explore_gain_ema + 0.2 * float(gain)

        if self._n_explore_trials < 3:
            return  # 样本太少, 先观察

        win_rate = self._n_explore_wins / self._n_explore_trials
        if win_rate >= 0.5 and self._explore_gain_ema > 0:
            # 探索有效 → 逐步升频
            self._explore_rate = min(self._explore_rate_max, self._explore_rate * 1.3)
        else:
            # 探索无效 → 逐步降频 (但保留最低探索量, 保证仍能发现新解)
            self._explore_rate = max(self._explore_rate_min, self._explore_rate * 0.7)

        # 每 4 次探索重置统计窗口, 避免历史结果永久锁死预算
        if self._n_explore_trials >= 4:
            self._n_explore_trials = 0
            self._n_explore_wins = 0

    def plan_action(self) -> Tuple[int, PolicyState]:
        """提出一个候选策略, 由**探索预算**决定是探索还是沿用 champion。

        为什么必须有探索门控:
          对照实验显示, 如果每次更新都换一组 bandit 采样出来的超参,
          训练配置每轮抖动, 线上性能反而不如固定超参 (E3 < E2)。
          正确的受控做法是:
            * 多数轮次 (1 - exploration_rate) 沿用 champion → 性能稳定
            * 少数轮次探索 challenger → 收集反馈, 有改进才晋升
          这正是"受控递归自改进"的工程含义: 探索必须受预算约束。
        """
        rate = float(self.cfg.get("policy.exploration_rate", 0.35))
        self._exploring = self._rng.random() < rate and not self._champion_locked
        if not self._exploring:
            return self._champion_arm_index, self.champion
        idx, challenger = self.bandit.select()
        return idx, challenger

    @property
    def _champion_arm_index(self) -> int:
        return sum(
            pb.values.index(getattr(self.champion, k))
            for k, pb in self.bandit.params.items()
            if getattr(self.champion, k) in pb.values
        )

    def candidate_overrides(self, policy: PolicyState, base: str) -> Dict[str, Any]:
        """Translate policy parameters into model hyper-parameter overrides.

        The policy layer searches over `lr` / `num_leaves` / `retrain_window`.
        Each base model needs its own translation — otherwise the bandit
        explores a dimension the model does not even read, and every
        candidate ends up identical (no learning signal, pure wasted compute).

        Model-specific translations:
          * ``lgbm``  -> learning_rate / num_leaves (native LightGBM names)
          * ``mlp``   -> learning_rate_init / hidden width
            (`num_leaves` maps to the hidden-layer width: 15/31/63 -> 64/128/256,
            which is exactly the capacity sweep that mattered on the RUL task)
          * ``rf``    -> max_depth (derived from num_leaves)

        ``retrain_window`` is handled separately (it controls how much data
        is assembled, not a model hyper-parameter).
        """
        if base == "lgbm":
            return {"learning_rate": policy.lr, "num_leaves": policy.num_leaves}
        if base == "mlp":
            # num_leaves (15/31/63) -> hidden width (64/128/256).
            # The RUL sweep showed capacity is the dominant factor:
            # 128 -> 539 RMSE, 256 -> 522 RMSE, 512 -> 618 RMSE (overfit).
            width = {15: 64, 31: 128, 63: 256}.get(
                int(policy.num_leaves), 128
            )
            return {
                "learning_rate_init": policy.lr,
                "hidden_layer_sizes": (width,),
            }
        if base == "rf":
            return {"max_depth": max(3, policy.num_leaves // 4)}
        return {}

    # -- 反馈 ------------------------------------------------------------
    def record(self, action: Action) -> None:
        self.budget.gpu_seconds += action.cost_gpu_seconds
        self.budget.labels += action.cost_labels
        self.budget.tokens += action.cost_tokens
        if not action.attempted:
            # 仅监控的轮次不产生策略反馈, 不应影响熔断计数与 bandit 统计
            return

        # 冠军-挑战者: 胜出的挑战者若**影子得分更高**, 晋升为新 champion。
        # 注意用影子得分而非 efficiency: efficiency 是"增益/成本"的比值,
        # 会被成本项扭曲, 不能直接用来判断"哪个超参更好"。
        if action.approved and action.gain > 0:
            if action.gain > self.champion_gain:
                self.champion = action.policy
                self.state = action.policy
                self.champion_gain = action.gain
            self.consecutive_failures = 0
        else:
            # 失败: 回退到 champion, 保证线上策略不抖
            self.state = self.champion
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.halted = True
                self._champion_locked = True  # 熔断后不再探索
                self.audit_note = (
                    f"连续 {self.consecutive_failures} 次更新失败, 熔断为只监控模式"
                )

    def feedback_candidate(
        self,
        variant: PolicyState,
        arm_index: int,
        reward: float,
        was_champion: bool = False,
        baseline: Optional[PolicyState] = None,
    ) -> None:
        """把**单个候选**的表现反馈给 bandit。

        在"多候选影子比选"机制下, 每轮会训练多个变体, 需要知道是哪个变体
        表现好 —— 否则 bandit 无法学到参数与收益的对应关系。

        `reward` 是**影子集上的绝对得分** (0~1), 只用于 bandit 的 Beta 后验;
        它与 `champion_efficiency` (单位成本增益) 是两个不同量纲, 不能混用。
        """
        if was_champion:
            # champion 的表现不构成对某个探索臂的检验。
            # 注意: 这里**不能**用它去覆盖 champion_efficiency ——
            # champion_efficiency 的语义是"单位成本增益", 与绝对得分不同量纲。
            return
        self.bandit.feedback(variant, arm_index, reward, baseline=baseline)

    def summary(self) -> Dict[str, Any]:
        return {
            "policy": self.state.to_dict(),
            "champion": self.champion.to_dict(),
            "champion_efficiency": round(self.champion_efficiency, 8),
            "champion_gain": round(self.champion_gain, 8),
            "exploring_locked": self._champion_locked,
            "budget": self.budget.to_dict(),
            "halted": self.halted,
            "consecutive_failures": self.consecutive_failures,
            "bandit": self.bandit.summary(),
            "n_actions": len(self.bandit.history),
        }
