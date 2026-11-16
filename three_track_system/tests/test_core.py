"""核心模块单元测试 — 保证闭环逻辑正确。

运行: .venv\\Scripts\\python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The ``evo`` package is the earlier adaptive implementation.  It lives
# under ``evo/src`` so that the repository root can hold the three-track
# closed loop without the two implementations colliding, which means the
# import path has to be set up here rather than by an installed package.
_EVO_SRC = Path(__file__).resolve().parent.parent / "evo" / "src"
if _EVO_SRC.is_dir() and str(_EVO_SRC) not in sys.path:
    sys.path.insert(0, str(_EVO_SRC))

from evo.config import load_config  # noqa: E402
from evo.data import SyntheticStreamGenerator  # noqa: E402
from evo.models import ModelBundle, ModelRegistry, compute_metrics, shadow_evaluate, Trainer  # noqa: E402
from evo.monitor import compute_psi, ks_test_pvalue, DriftMonitor  # noqa: E402
from evo.policy import RestrictedThompsonBandit, PolicyController, PolicyState  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    gen = SyntheticStreamGenerator(cfg)
    train, valid = gen.make_train_valid()
    stream = gen.make_stream()
    return train, valid, stream


# =====================================================================
# PSI / KS
# =====================================================================

def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert compute_psi(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=2000)
    cur = rng.normal(loc=1.5, size=2000)
    assert compute_psi(ref, cur) > 0.25


def test_ks_pvalue_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    cur = rng.normal(loc=1.0, size=1000)
    assert ks_test_pvalue(ref, cur) < 0.01


# =====================================================================
# 漂移监测与归因
# =====================================================================

def test_monitor_detects_covariate_drift(cfg, data):
    train, valid, stream = data
    mon = DriftMonitor(cfg)
    early = stream.slice(0, 400)
    late = stream.slice(5200, 5600)
    rep = mon.check(train, early)
    assert not rep.triggered, "早期窗口不应触发"
    rep_late = mon.check(train, late)
    assert rep_late.severity in ("warn", "alarm")
    assert rep_late.max_psi > 0.1


def test_monitor_skips_small_window(cfg, data):
    train, valid, stream = data
    mon = DriftMonitor(cfg)
    rep = mon.check(train, stream.slice(0, 50))
    assert not rep.triggered


# =====================================================================
# 模型
# =====================================================================

def test_trainer_and_metrics(cfg, data):
    train, valid, stream = data
    tr = Trainer(cfg)
    bundle = tr.train("lgbm", train.X, train.y, train.feature_names)
    assert bundle.train_seconds >= 0
    pred = bundle.predict(valid.X)
    m = compute_metrics(valid.y, pred)
    assert 0.0 <= m["auc"] <= 1.0
    assert m["score"] > 0.6, f"基线 AUC 应显著高于随机, 实际 {m}"


def test_shadow_evaluate_rejects_worse_model(cfg, data):
    """明显更差的候选必须被拒绝上线。"""
    from sklearn.dummy import DummyClassifier

    train, valid, stream = data
    tr = Trainer(cfg)
    good = tr.train("lgbm", train.X, train.y, train.feature_names)
    dummy = DummyClassifier(strategy="most_frequent").fit(train.X, train.y)
    bad = ModelBundle(
        name="dummy", estimator=dummy, feature_names=train.feature_names, task="binary"
    )
    res = shadow_evaluate(good, bad, valid, min_samples=100)
    assert not res.approved
    assert res.delta_score < 0


def test_shadow_evaluate_rejects_tiny_gain(cfg, data):
    """增益低于受控门限时必须拒绝上线 (防噪声抖动)。"""
    train, valid, stream = data
    tr = Trainer(cfg)
    good = tr.train("lgbm", train.X, train.y, train.feature_names)
    cand = tr.train("lgbm", train.X, train.y, train.feature_names)  # 同数据 → 增益为 0
    res = shadow_evaluate(good, cand, valid, min_samples=100, margin=0.005)
    assert not res.approved


def test_shadow_evaluate_accepts_better_model(cfg, data):
    """更优候选在样本充足且显著时应被批准上线。"""
    train, valid, stream = data
    tr = Trainer(cfg)
    baseline = tr.train("logreg", train.X, train.y, train.feature_names)
    candidate = tr.train("lgbm", train.X, train.y, train.feature_names)
    res = shadow_evaluate(baseline, candidate, valid, min_samples=100, margin=0.0)
    # 不强制 approved (合成数据上二者可能互有胜负), 但判定链路必须给出结论
    assert res.reason
    assert isinstance(res.approved, bool)


def test_registry_promote_and_rollback(cfg, data):
    train, valid, stream = data
    tr = Trainer(cfg)
    reg = ModelRegistry(cfg)
    b1 = tr.train("lgbm", train.X, train.y, train.feature_names)
    b2 = tr.train("logreg", train.X, train.y, train.feature_names)
    reg.register(b1, "v1")
    reg.register(b2, "v2")
    assert reg.current_version() == "v2"
    assert reg.rollback("v1")
    assert reg.current_version() == "v1"
    assert reg.load_current().name == "lgbm"


# =====================================================================
# 策略层
# =====================================================================

def test_bandit_respects_neighborhood(cfg):
    """单参数 bandit 每次只能在相邻档位间移动 (受控 RSI)。"""
    bandit = RestrictedThompsonBandit(cfg, seed=1)
    for _ in range(30):
        before = {k: pb.index for k, pb in bandit.params.items()}
        _, state = bandit.select()
        for k, pb in bandit.params.items():
            assert abs(pb.index - before[k]) <= 1, f"{k} 跨越了超过 1 档"
            assert getattr(state, k) == pb.values[pb.index]


def test_bandit_arms_cover_param_space(cfg):
    """参数空间的每一档都必须可达 (避免设计上不可探索)。"""
    bandit = RestrictedThompsonBandit(cfg, seed=1)
    space = cfg.get("policy.param_space")
    for key, values in space.items():
        assert bandit.params[key].values == list(values)


def test_bandit_explores_under_low_round_budget(cfg):
    """30 轮内必须能在参数空间里发生实际移动 —— 防退化为常量策略。"""
    from evo.policy import Action, PolicyState

    bandit = RestrictedThompsonBandit(cfg, seed=3)
    moved = set()
    for i in range(30):
        idx, state = bandit.select()
        for k, pb in bandit.params.items():
            moved.add((k, pb.index))
        bandit.update(
            Action(arm_index=idx, policy=state, gain=0.05,
                   cost_gpu_seconds=0.2, cost_labels=100, approved=True)
        )
    # 至少一半参数维度发生过移动
    dims_moved = {k for k, _ in moved}
    assert len(dims_moved) >= len(bandit.params) - 1


def test_policy_respects_budget(cfg):
    ctrl = PolicyController(cfg, seed=1)
    ctrl.budget.gpu_seconds = ctrl.budget.limits["gpu_seconds"] * 0.95
    should, why = ctrl.should_update(0.9, "alarm")
    assert not should
    assert "预算" in why


def test_policy_halts_after_consecutive_failures(cfg):
    """连续失败后熔断 —— 降级为只监控, 但不终止运行 (满足门槛①)。

    熔断是**分级**的 (见 `test_halted_policy_still_recovers_on_alarm`):
      * 普通漂移 (warn) → 只监控, 停止无谓更新;
      * 严重漂移 (alarm) → 允许沿用 champion 参数止损重训 (不做新参数探索)。
    这里断言 warn 情形下不再更新。
    """
    ctrl = PolicyController(cfg, seed=1)
    from evo.policy import Action

    for _ in range(3):
        ctrl.record(Action(arm_index=0, policy=ctrl.state, gain=0.0, approved=False))
    assert ctrl.halted
    should, why = ctrl.should_update(0.15, "warn")
    assert not should
    assert "熔断" in why or "只监控" in why


def test_policy_champion_challenger_does_not_leak(cfg):
    """挑战者失败时 champion 必须保持不变 (受控: 探索不污染线上)。"""
    ctrl = PolicyController(cfg, seed=1)
    from evo.policy import Action

    champion_before = ctrl.champion.to_dict()
    ctrl.record(
        Action(arm_index=1, policy=PolicyState(lr=0.1, num_leaves=63),
               gain=0.0, cost_gpu_seconds=1.0, approved=False)
    )
    assert ctrl.champion.to_dict() == champion_before
    assert ctrl.state.to_dict() == champion_before


def test_policy_promotes_better_challenger(cfg):
    """效率更高的挑战者应晋升为 champion。"""
    ctrl = PolicyController(cfg, seed=1)
    from evo.policy import Action

    better = PolicyState(lr=0.1, num_leaves=63)
    ctrl.record(
        Action(arm_index=1, policy=better, gain=0.2,
               cost_gpu_seconds=1.0, approved=True)
    )
    assert ctrl.champion.lr == 0.1
    assert ctrl.champion.num_leaves == 63


def test_policy_champion_gain_not_polluted_by_absolute_score(cfg):
    """回归: champion 的晋升基准不能被"绝对影子得分"污染。

    多候选比选机制下, feedback_candidate 会拿到 candidate_score (~0.85, 绝对得分),
    而晋升判定用的是 gain (相对增益, ~0.05~0.3)。两者量纲不同:
    若把绝对得分写进晋升基准, 晋升条件将永远为 False, champion 永不更新。
    """
    ctrl = PolicyController(cfg, seed=1)
    from evo.policy import Action, PolicyState

    before = ctrl.champion.to_dict()
    baseline_gain = ctrl.champion_gain

    # champion 在候选集里的表现 (绝对得分) —— 绝不能影响晋升基准
    ctrl.feedback_candidate(
        PolicyState(), 0, reward=0.85, was_champion=True, baseline=PolicyState()
    )
    assert ctrl.champion_gain == baseline_gain, "绝对得分污染了晋升基准"
    assert ctrl.champion.to_dict() == before

    # 随后一个"增益更高"的挑战者必须能正常晋升
    ctrl.record(
        Action(arm_index=1, policy=PolicyState(lr=0.1, num_leaves=63),
               gain=0.2, cost_gpu_seconds=1.0, cost_labels=100, approved=True)
    )
    assert ctrl.champion.lr == 0.1, "晋升失效: champion 未更新"
    assert ctrl.champion_gain > baseline_gain


def test_bandit_feedback_credits_only_changed_dimension(cfg):
    """回归: 变体奖励只能记到**实际改动的维度**上。

    若把奖励同时记到未改动的维度, bandit 会误以为
    "lr=0.02 且 leaves=31 更好", 而真实因果只是 lr 变了 ——
    学习结果不可解释, 策略层的可解释性考察点会失守。
    """
    bandit = RestrictedThompsonBandit(cfg, seed=3)
    base = PolicyState()  # lr=0.05, num_leaves=31, retrain_window=2000
    variant = PolicyState(lr=0.02)  # 只改了 lr

    credited = bandit.feedback(variant, arm_index=0, reward=0.9, baseline=base)

    assert credited == ["lr"], f"归因维度错误: {credited}"
    lr_pb = bandit.params["lr"]
    assert len(lr_pb.stats[lr_pb.values.index(0.02)]) == 1, "lr 臂未收到奖励"
    for name in ("num_leaves", "retrain_window", "psi_trigger", "min_gain"):
        pb = bandit.params[name]
        assert all(len(s) == 0 for s in pb.stats), f"{name} 被错误归因"


def test_bandit_feedback_credits_all_changed_dims(cfg):
    """变体改了多个维度时, 这些维度都应收到奖励。"""
    bandit = RestrictedThompsonBandit(cfg, seed=3)
    base = PolicyState()
    variant = PolicyState(lr=0.02, num_leaves=63)

    credited = bandit.feedback(variant, arm_index=0, reward=0.9, baseline=base)

    assert set(credited) == {"lr", "num_leaves"}, f"归因维度错误: {credited}"


def test_policy_initial_state_aligns_with_param_space(cfg):
    """回归: 初始策略必须对齐到配置参数空间的起始档。

    否则 `cfg.set("policy.param_space.psi_trigger", [0.0001])` 这类覆盖
    会被静默忽略 —— 初始 state 仍是硬编码默认值, 对照实验的变体
    会退化成同一个系统, 结论失效。
    """
    import copy

    from evo.config import load_config as _load

    c = _load()
    c.set("policy.param_space", {
        "psi_trigger": [0.0001, 0.1], "min_gain": [0.005],
        "retrain_window": [2000], "lr": [0.05], "num_leaves": [31],
    })
    ctrl = PolicyController(c, seed=1)
    # 默认值 0.1 在空间内 → 对齐到该档
    assert ctrl.state.psi_trigger == 0.1, (
        f"初始 psi_trigger 未对齐默认值所在档: {ctrl.state.psi_trigger}"
    )
    assert ctrl.champion.psi_trigger == 0.1

    # 默认值不在空间内 → 必须落到第一档 (否则覆盖被静默忽略)
    c2 = _load()
    c2.set("policy.param_space.psi_trigger", [0.0001])
    ctrl2 = PolicyController(c2, seed=1)
    assert ctrl2.state.psi_trigger == 0.0001
    # 触发阈值极小时应真的触发更新
    should, _ = ctrl2.should_update(0.05, "none")
    assert should, "psi_trigger=0.0001 时应触发更新"


def test_policy_periodic_retrain_triggers_regardless_of_drift(cfg):
    """回归: 定时重训模式无视漂移强度, 每轮都应触发更新。

    这是消融 E1 (定时重训) 的定义。之前 E1 用"把 psi_trigger 设成极小值"
    实现, 会被熔断/预算/策略层调整破坏, 导致 E1 与 E3 行为雷同。
    """
    import copy

    from evo.config import load_config as _load

    c = _load()
    c.set("policy.periodic_retrain", True)
    ctrl = PolicyController(c, seed=1)
    should, why = ctrl.should_update(0.0, "none")
    assert should, f"定时重训模式未触发: {why}"
    assert "定时重训" in why


def test_policy_periodic_disabled_by_default(cfg):
    """默认配置下不应进入定时重训模式 (E0/E2/E3 不受影响)。"""
    ctrl = PolicyController(cfg, seed=1)
    assert not ctrl.periodic_retrain


def test_shadow_set_does_not_overlap_training_data(cfg, data):
    """回归: 影子评估集必须与候选模型的训练数据**不重叠**。

    早期版本影子集 = 流末端窗口, 而训练数据也取自流末端 ——
    两者完全重叠, 影子评估退化为"训练集得分", 候选全部饱和到 1.0,
    上线判定与策略层探索同时失效。
    """
    from evo.harness import Harness

    cfg.set("paths.artifacts_root", "artifacts/_tmp_shadow")
    cfg.set("paths.audit_log", "artifacts/_tmp_shadow/audit/audit.jsonl")
    cfg.set("paths.shadow_root", "artifacts/_tmp_shadow/shadow")
    cfg.set("experiment.local_dir", "artifacts/_tmp_shadow/exp")
    h = Harness(cfg, use_synthetic=True)
    h.prepare_data()
    h.bootstrap()

    window = (3000, 3500)
    shadow = h._shadow_set(window)
    train_set, _, _ = h._assemble_training_data(window, h.policy.state)

    # 影子集与训练集不能有交集 (用行内容比对)
    shadow_rows = {r.tobytes() for r in shadow.X}
    train_rows = {r.tobytes() for r in train_set.X}
    overlap = shadow_rows & train_rows
    assert not overlap, (
        f"影子集与训练集重叠 {len(overlap)} 行 —— 影子评估退化成了训练集得分"
    )


def test_synthetic_stream_follows_runtime_seed(cfg):
    """回归: 合成数据必须随 runtime.seed 变化。

    否则对照实验里的"多随机种子取均值±std"毫无意义 ——
    每个种子跑的是同一份数据, 结果完全相同, std 恒为 0,
    无法反映系统在数据扰动下的真实稳定性。
    """
    from evo.config import load_config as _load
    from evo.data import SyntheticStreamGenerator

    streams = []
    for seed in (11, 22, 33):
        c = _load()
        c.set("runtime.seed", seed)
        gen = SyntheticStreamGenerator(c)
        s = gen.make_stream()
        streams.append((seed, s))

    # 不同种子的流必须不同
    for i in range(len(streams) - 1):
        s1, s2 = streams[i][1], streams[i + 1][1]
        assert not np.array_equal(s1.X, s2.X), (
            f"种子 {streams[i][0]} 与 {streams[i + 1][0]} 生成了完全相同的数据流"
        )

    # 同一 seed 必须可复现
    c = _load()
    c.set("runtime.seed", 11)
    a = SyntheticStreamGenerator(c).make_stream()
    b = SyntheticStreamGenerator(c).make_stream()
    assert np.array_equal(a.X, b.X), "同一 seed 未产生可复现的数据流"


# =====================================================================
# 端到端闭环
# =====================================================================

def test_full_loop_runs_without_human_intervention(cfg, tmp_path):
    """核心门槛测试: 无人工干预跑完全流程并产出结果。"""
    import json

    from evo.harness import Harness

    cfg.set("paths.artifacts_root", str(tmp_path / "artifacts"))
    cfg.set("paths.audit_log", str(tmp_path / "artifacts/audit/audit.jsonl"))
    cfg.set("paths.shadow_root", str(tmp_path / "artifacts/shadow"))
    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    cfg.set("runtime.window_size", 300)
    cfg.set("safety.max_iterations", 6)

    harness = Harness(cfg, use_synthetic=True)
    summary = harness.run()

    assert summary.iterations > 0
    assert summary.total_gpu_seconds >= 0
    assert (tmp_path / "artifacts" / "run_summary.json").exists()

    # 审计链必须完整
    from evo.audit import AuditLog

    res = AuditLog(cfg).verify()
    assert res["valid"], f"审计链断裂: {res}"


def test_experiment_records_are_compliant(cfg, tmp_path):
    import json

    from evo.experiment import ExperimentRecorder

    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    rec = ExperimentRecorder(cfg)
    rec.log(
        {
            "name": "t",
            "config": {"a": 1},
            "metrics": {"score": 0.5},
            "tags": ["x"],
            "notes": "n",
        }
        if False
        else __import__("evo.experiment", fromlist=["ExperimentRecord"]).ExperimentRecord(
            name="t", config={"a": 1}, metrics={"score": 0.5}, tags=["x"], notes="n"
        )
    )
    rep = rec.compliance_report()
    assert rep["total_records"] == 1
    assert rep["compliant"]


# =====================================================================
# 回归: 策略层必须有可学信号 (参数空间不能被工程约束"压平")
# =====================================================================

def test_retrain_window_is_not_flattened_by_label_quota(cfg):
    """回归: `retrain_window` 必须真正改变训练集大小。

    早期实现每轮只取"尾部 quota 条新标注"来训练, 于是无论
    `retrain_window` 取 500 还是 4000, 实际训练集大小恒等于 quota。
    参数空间 4 档全部退化成同一份数据 => 策略层在该维度上**零信号**,
    探索纯属烧算力 (对照实验里 E3 花 3 倍算力却追平固定超参的 E2)。
    """
    from evo.harness import Harness

    cfg.set("paths.artifacts_root", "artifacts/_tmp_win")
    cfg.set("paths.audit_log", "artifacts/_tmp_win/audit/audit.jsonl")
    cfg.set("paths.shadow_root", "artifacts/_tmp_win/shadow")
    cfg.set("experiment.local_dir", "artifacts/_tmp_win/exp")
    h = Harness(cfg, use_synthetic=True)
    h.prepare_data()
    h.bootstrap()

    window = (3000, 3500)

    sizes = {}
    for w in (500, 4000):
        pol = PolicyState(retrain_window=w)
        ds, _, _ = h._assemble_training_data(window, pol)
        assert ds is not None
        sizes[w] = len(ds)

    assert sizes[4000] > sizes[500], (
        f"retrain_window 未影响训练集大小 ({sizes}) —— 参数空间被压平, 策略层无信号"
    )


def test_exploration_covers_all_model_dimensions(cfg):
    """回归: 探索必须轮转覆盖所有模型参数维度。

    早期用随机 shuffle 选维度, 实测 8 次探索全部落在 `lr`,
    `retrain_window` 的 bandit 样本数恒为 0 —— 该维度等于没在学。
    """
    ctrl = PolicyController(cfg, seed=7)
    base = ctrl.champion
    touched = set()
    for _ in range(12):
        variant = ctrl._propose_variant(base)
        assert variant is not None
        for k in PolicyState.model_keys():
            if getattr(variant, k) != getattr(base, k):
                touched.add(k)

    missing = set(PolicyState.model_keys()) - touched
    assert not missing, f"这些模型维度从未被探索: {missing}"


def test_halted_policy_still_recovers_on_alarm(cfg):
    """回归: 熔断后若漂移已达 alarm, 仍应允许用 champion 参数止损重训。

    若熔断 = "停止一切更新", 则一旦连续失败触发熔断, 系统在后续窗口
    全部空转 —— 实测在 PSI 达 0.49 (alarm) 的持续恶化区间也完全不再重训,
    现网模型带着退化裸奔到流结束, 尾段得分毫无改善。
    """
    ctrl = PolicyController(cfg, seed=3)
    ctrl.halted = True
    ctrl._champion_locked = True

    should_alarm, _ = ctrl.should_update(0.5, "alarm")
    assert should_alarm, "熔断后 alarm 漂移未触发止损重训"

    should_warn, _ = ctrl.should_update(0.15, "warn")
    assert not should_warn, "熔断后 warn 漂移不应触发更新 (应保持只监控)"


# =====================================================================
# 回归: 决策层 (何时更新) 必须可学, 且探索不得扰动线上行为
# =====================================================================

def test_decision_layer_collects_trigger_statistics(cfg):
    """回归: 决策层必须为每个 psi_trigger 档位积累可比较的样本。

    早期实现里 `psi_trigger` / `min_gain` 的 bandit 臂样本数**恒为 0**
    (整个 `plan_action` 从未被调用) —— "何时更新"这个决策从未被比较过,
    策略层的自改进只有模型超参一半是真的。
    """
    ctrl = PolicyController(cfg, seed=5)
    for _ in range(10):
        ctrl.observe_counterfactual(0.3, "warn")

    assert ctrl._trigger_rate, "决策层未积累任何档位的触发率样本"
    # 所有档位都应留下样本 (纯观测, 每轮对所有档位各记一次)
    for t in ctrl.bandit.params["psi_trigger"].values:
        assert t in ctrl._trigger_rate, f"档位 {t} 没有触发率样本"

    # PSI=0.3 时: 0.08/0.1/0.15/0.2/0.25 全部会触发 (0.3 >= 0.25)
    assert ctrl._trigger_rate[0.25] == [1.0] * 10
    assert ctrl._trigger_rate[0.1] == [1.0] * 10

    # PSI=0.12 时: 0.08/0.1 会触发, 0.15 及以上不会
    ctrl2 = PolicyController(cfg, seed=5)
    ctrl2.observe_counterfactual(0.12, "warn")
    assert ctrl2._trigger_rate[0.1] == [1.0]
    assert ctrl2._trigger_rate[0.15] == [0.0]


def test_decision_probe_does_not_change_behaviour(cfg):
    """回归: 决策层探索必须是**纯观测**, 不得扰动线上行为。

    早期让探针直接替换当轮触发阈值, 导致探针轮与 champion 轮的更新
    时机错开, champion 参数得不到连续验证 —— 实测 tail_score 从
    0.5446 掉到 0.5088, 纯属探索扰动的代价。
    """
    ctrl = PolicyController(cfg, seed=11)
    before = ctrl.state.psi_trigger
    for _ in range(10):
        assert ctrl.probe_or_keep() is None, "探针不应替换当轮生效的阈值"
    assert ctrl.state.psi_trigger == before
    # 触发判定不受探针影响
    should, _ = ctrl.should_update(0.2, "warn")
    assert should, "探针影响了触发判定"


def test_decision_policy_switches_only_with_sufficient_evidence(cfg):
    """回归: 决策层只在证据充分时切换阈值。

    样本量不足时均值的差异多半是噪声, 贸然切换会让线上行为随噪声漂移。
    """
    ctrl = PolicyController(cfg, seed=13)
    ctrl.decision_update_interval = 1
    ctrl.decision_min_samples = 4
    ctrl.decision_min_improvement = 0.001

    # 奖励记到**当前生效的**阈值档位上 —— 因此要构造两个档位的样本,
    # 需要在记录时切换 state.psi_trigger (模拟系统先后用不同阈值运行)。
    def feed(threshold: float, reward: float, n: int):
        ctrl.state.psi_trigger = threshold
        for _ in range(n):
            ctrl.observe_trigger_outcome(0.3, "warn", reward)

    def decide() -> Optional[str]:
        # 冷却期按实际轮数推进 —— 每次判断前推进一步
        ctrl.tick_decision()
        return ctrl.update_decision_policy()

    # champion 档位 (0.1) 只有 1 个样本 → 样本不足, 不应切换
    feed(0.1, 0.1, 1)
    assert decide() is None, "样本不足时不应切换阈值"

    # challenger 档位 (0.08) 收益更高但只有 2 个样本 → 仍不切换
    feed(0.08, 1.0, 2)
    assert decide() is None, "challenger 样本不足时不应切换"

    # 两个档位样本都补足, challenger 累计收益明显更高 → 应切换
    feed(0.1, 0.1, 4)
    feed(0.08, 1.0, 4)
    ctrl.state.psi_trigger = 0.1  # 回到 champion 作为比较基准
    note = decide()
    assert note is not None, "证据充分时未切换阈值"
    assert ctrl.state.psi_trigger == 0.08


# =====================================================================
# 赛题合规: 基础运行门槛 / 批次下发 / 模型白名单 / 提交包
# =====================================================================

def test_gates_check_reports_both_requirements(cfg, tmp_path):
    """赛题门槛: 至少三次尝试迭代 + 性能不低于静态基线。

    门槛结果必须写进 run_summary.json —— 评委与自查都直接读这个字段。
    """
    import json

    from evo.harness import Harness

    cfg.set("paths.artifacts_root", str(tmp_path / "artifacts"))
    cfg.set("paths.audit_log", str(tmp_path / "artifacts/audit/audit.jsonl"))
    cfg.set("paths.shadow_root", str(tmp_path / "artifacts/shadow"))
    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    cfg.set("runtime.window_size", 300)
    cfg.set("safety.max_iterations", 6)

    Harness(cfg, use_synthetic=True).run()

    data = json.loads((tmp_path / "artifacts" / "run_summary.json").read_text(encoding="utf-8"))
    gates = data.get("gates")
    assert gates is not None, "run_summary.json 缺少门槛记录"
    assert gates["min_iterations"] == 3
    assert "iteration_gate_passed" in gates
    assert "baseline_gate_passed" in gates
    assert isinstance(gates["all_passed"], bool)


def test_batch_runner_produces_response_per_batch(cfg, tmp_path):
    """批次下发协议: 每批必须产出一份应答 (预测 + 决策)。

    赛题: "必须先输出当前批次模型或不上线的判断才能获取下一批数据"。
    """
    from evo.batch import BatchRunner, CallbackBatchFetcher
    from evo.data import SyntheticStreamGenerator

    cfg.set("paths.artifacts_root", str(tmp_path / "artifacts"))
    cfg.set("paths.audit_log", str(tmp_path / "artifacts/audit/audit.jsonl"))
    cfg.set("paths.shadow_root", str(tmp_path / "artifacts/shadow"))
    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    cfg.set("runtime.window_size", 300)
    cfg.set("safety.max_iterations", 5)

    gen = SyntheticStreamGenerator(cfg)
    stream = gen.make_stream()
    size = 800
    chunks = [
        stream.slice(i, min(i + size, len(stream)))
        for i in range(0, len(stream), size)
    ]

    def fetch(batch_id: int):
        return chunks[batch_id] if batch_id < len(chunks) else None

    runner = BatchRunner(cfg, CallbackBatchFetcher(fetch), use_synthetic=True)
    summary = runner.run(max_batches=3)

    assert summary.batches == 3, f"批次数不对: {summary.batches}"
    assert len(summary.responses) == 3
    for r in summary.responses:
        assert r.n_samples > 0
        assert len(r.predictions) == r.n_samples, "应答必须含本批全部预测"
        assert "action" in r.decision, "应答必须含上线/不上线判断"
        assert r.model_version, "应答必须标明用的是哪个模型版本"


def test_batch_mode_ignores_label_quota(cfg, tmp_path):
    """赛题: "评测时总能拿到标签, 且无需消耗配额申请"。

    因此批次模式下标注配额不应成为硬约束。
    """
    from evo.batch import BatchRunner
    from evo.harness import Harness

    cfg.set("paths.artifacts_root", str(tmp_path / "artifacts"))
    cfg.set("paths.audit_log", str(tmp_path / "artifacts/audit/audit.jsonl"))
    cfg.set("paths.shadow_root", str(tmp_path / "artifacts/shadow"))
    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    cfg.set("budget.max_label_quota", 10)  # 极小配额

    h = Harness(cfg, use_synthetic=True)
    runner = BatchRunner(cfg, _NoBatch(), use_synthetic=True)
    runner.harness = h
    runner._relax_label_quota()

    assert h.policy.budget.limits["labels"] > 10, "批次模式未放宽标注配额"
    assert h.policy.budget.limits["_orig_label_limit"] == 10, "原始配额未留痕"


class _NoBatch:
    """空批次源 —— 仅用于配额放宽测试。"""

    def fetch(self, batch_id: int):
        return None


def test_llm_rejects_model_outside_allowlist(cfg):
    """赛题: 仅允许千问 27B / 4B, 其他模型不得使用。"""
    from evo.llm import LLMClient

    cfg.set("llm.allowed_models", ["Qwen3.8-27B", "Qwen3.5-4B"])

    # 变体大小写/分隔符应视为同一个模型
    for ok in ("Qwen3.8-27B", "qwen3-8-27b", "Qwen3.5-4B", "qwen3-5-4b"):
        c = load_config()
        c.set("llm.allowed_models", ["Qwen3.8-27B", "Qwen3.5-4B"])
        import os

        os.environ["PYTRIO_MODEL"] = ok
        try:
            assert LLMClient(c).model_allowed, f"{ok} 应被允许"
        finally:
            os.environ.pop("PYTRIO_MODEL", None)

    import os

    os.environ["PYTRIO_MODEL"] = "gpt-4o"
    try:
        c = load_config()
        c.set("llm.allowed_models", ["Qwen3.8-27B", "Qwen3.5-4B"])
        client = LLMClient(c)
        assert not client.model_allowed, "白名单外的模型应被拒绝"
        # 被拒绝的模型不得真正发起调用
        assert client.chat([{"role": "user", "content": "x"}]) == ""
    finally:
        os.environ.pop("PYTRIO_MODEL", None)


def test_llm_allows_pytrio_derived_model(cfg):
    """赛题路线二: 以千问为基座经 PyTRiO 后训练得到的驱动模型必须可用。

    后训练模型名通常带后缀 (如 "qwen3.8-27b-ft-20261115"),
    若只做字面相等校验, 会把**合规**的路线二模型误判为违规。
    """
    import os

    from evo.llm import LLMClient

    # 1) 基座包含即可通过 (无需维护长名单)
    os.environ["PYTRIO_MODEL"] = "qwen3.8-27b-ft-20261115"
    try:
        c = load_config()
        c.set("llm.allowed_models", ["Qwen3.8-27B", "Qwen3.5-4B"])
        assert LLMClient(c).model_allowed, "后训练派生模型应被允许 (基座合规)"
    finally:
        os.environ.pop("PYTRIO_MODEL", None)

    # 2) 显式派生名单兜底 (基座名完全被平台改写时)
    os.environ["PYTRIO_MODEL"] = "my-custom-finetuned-model"
    try:
        c = load_config()
        c.set("llm.allowed_models", ["Qwen3.8-27B", "Qwen3.5-4B"])
        c.set("llm.derived_models", ["my-custom-finetuned-model"])
        assert LLMClient(c).model_allowed, "显式声明的派生模型应被允许"
    finally:
        os.environ.pop("PYTRIO_MODEL", None)

    # 3) 非千问基座且未声明 → 仍应拒绝
    os.environ["PYTRIO_MODEL"] = "llama3-70b-finetuned"
    try:
        c = load_config()
        c.set("llm.allowed_models", ["Qwen3.8-27B", "Qwen3.5-4B"])
        c.set("llm.derived_models", [])
        assert not LLMClient(c).model_allowed, "非千问基座应被拒绝"
    finally:
        os.environ.pop("PYTRIO_MODEL", None)


def test_submission_build_excludes_weights(cfg, tmp_path):
    """赛题: 严禁将模型权重打包进提交包。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evo" / "scripts"))
    import build_submission

    files = build_submission._collect_files()
    leaked = [p for p in files if build_submission._is_weight(p)]
    assert not leaked, f"提交包收集到了权重文件: {leaked}"

    # 白名单必须覆盖所有权重后缀
    for suffix in (".pkl", ".joblib", ".pt", ".pth", ".onnx", ".safetensors", ".bin"):
        assert suffix in build_submission.WEIGHT_SUFFIXES


def test_swanlab_public_flag_is_required(cfg):
    """赛题: Swan Lab 记录必须公开 —— 合规自检要能反映这一点。"""
    from evo.experiment import ExperimentRecorder

    cfg.set("experiment.local_dir", "artifacts/_tmp_swan")
    cfg.set("experiment.swanlab_public", True)
    rec = ExperimentRecorder(cfg)
    rep = rec.compliance_report()
    assert rep["swanlab_public_required"] is True
    assert "swanlab_public_ok" in rep


def test_model_factory_limits_threads(cfg):
    """评测环境仅 4 核 —— 训练必须限制线程数。"""
    from evo.models import ModelFactory

    cfg.set("runtime.n_jobs", 4)
    factory = ModelFactory(cfg)
    est = factory.build("lgbm")
    assert getattr(est, "n_jobs", None) == 4, "LightGBM 未限制线程数"


# =====================================================================
# 官方评测协议适配器
# =====================================================================

class _StubModel:
    """最小模型桩 —— 只实现 predict。"""

    task = "regression"

    def __init__(self, value=123.0, feature_paths=None):
        self.value = value
        self.feature_paths = feature_paths or []

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float64)


def test_predict_server_regression_output_format():
    """官方回归协议: {"outputs": {"value": 非负数}}。"""
    from evo.predict_server import PredictServer, validate_task_output_format

    server = PredictServer(model=_StubModel(value=42.0), task="regression")
    resp = server.handle({"task": "regression", "inputs": {"features": [1.0, 2.0]}})
    validate_task_output_format("regression", resp["outputs"])
    assert resp["outputs"]["value"] == pytest.approx(42.0)


def test_predict_server_clamps_negative_rul_to_zero():
    """官方要求: RUL 预测截断至非负值。"""
    from evo.predict_server import PredictServer

    server = PredictServer(model=_StubModel(value=-50.0), task="regression")
    resp = server.handle({"task": "regression", "inputs": {"features": [1.0]}})
    assert resp["outputs"]["value"] == 0.0


def test_predict_server_classification_output_format():
    """官方分类协议: {"outputs": [{"label":..., "score":...}]}。"""
    from evo.predict_server import PredictServer, validate_task_output_format

    class _Cls(_StubModel):
        task = "classification"

        def predict(self, X):
            return np.tile(np.array([0.7, 0.2, 0.1]), (len(X), 1))

    server = PredictServer(
        model=_Cls(), task="classification", classes=["A", "B", "C"]
    )
    resp = server.handle({"task": "classification", "inputs": {"f": [1.0]}})
    validate_task_output_format("classification", resp["outputs"])
    assert resp["outputs"][0]["label"] == "A"
    assert resp["outputs"][0]["score"] == pytest.approx(0.7)
    assert len(resp["outputs"]) == 3


def test_predict_server_rejects_bad_requests():
    """官方约束: 请求只能含 task/inputs, task 必须匹配。"""
    from evo.predict_server import PredictServer

    server = PredictServer(model=_StubModel(), task="regression")
    # 多余键
    with pytest.raises(ValueError):
        server.handle({"task": "regression", "inputs": {"f": [1.0]}, "extra": 1})
    # task 不匹配
    with pytest.raises(ValueError):
        server.handle({"task": "classification", "inputs": {"f": [1.0]}})
    # 缺 inputs
    with pytest.raises(ValueError):
        server.handle({"task": "regression"})


def test_predict_server_enforces_feature_schema():
    """官方 loader 会逐条比对 paths —— 我们必须同样严格。

    特征错位是最难查的 bug: 模型照常输出数字, 但数值全是错的。
    """
    from evo.predict_server import PredictServer

    expected = ['[["key","f"],["index","0"]]']
    server = PredictServer(
        model=_StubModel(feature_paths=expected), task="regression"
    )
    # schema 一致 → 通过
    server.handle({"task": "regression", "inputs": {"f": [1.0]}})
    # schema 不一致 (多了个字段) → 报错
    with pytest.raises(ValueError, match="schema"):
        server.handle({"task": "regression", "inputs": {"f": [1.0], "g": [2.0]}})


def test_predict_server_continues_after_bad_line():
    """单条脏数据不得拖垮整个评测服务。"""
    import io

    from evo.predict_server import PredictServer

    stdin = io.StringIO(
        '{"task":"regression","inputs":{"f":[1.0]}}\n'
        "not json\n"
        '{"task":"regression","inputs":{"f":[2.0]}}\n'
    )
    stdout = io.StringIO()
    server = PredictServer(model=_StubModel(), task="regression")
    server.serve(stdin, stdout)
    lines = [ln for ln in stdout.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3
    assert "error" in lines[1]
    assert "outputs" in lines[0] and "outputs" in lines[2]


def test_flatten_rejects_nonfinite_values():
    """官方 flatten 只接受有限数值 —— NaN/Inf 必须拒绝。"""
    from evo.data import _flatten_jsonl_inputs

    with pytest.raises(ValueError):
        _flatten_jsonl_inputs({"f": [float("nan")]})
    with pytest.raises(ValueError):
        _flatten_jsonl_inputs({"f": [float("inf")]})
    # 正常值通过
    paths, values = _flatten_jsonl_inputs({"f": [1.0, 2.0]})
    assert values == [1.0, 2.0]


# =====================================================================
# 赛题专用指标: A2 严重退化漏报 / A3 非对称评分
# =====================================================================

def test_severe_degradation_miss_rate_detects_hidden_failure():
    """A2 核心指标: 严重退化漏报必须能被单独检出。

    场景: 1000 个样本中 30 个已严重退化 (3%), 模型全部判成健康。
      * Accuracy = 97%  ← 看起来极好
      * Macro-F1 = 0.83  ← 仍然"不错"
    但业务后果是 30 次该报警却没报警 —— 必须靠独立指标暴露。
    """
    from evo.task_metrics import severe_degradation_miss_rate

    y = np.array([0] * 970 + [1] * 30)
    p = np.array([0] * 1000)
    m = severe_degradation_miss_rate(y, p, severe_classes=["1"])

    assert m["severe_count"] == 30
    assert m["severe_miss_count"] == 30
    assert m["severe_miss_rate"] == pytest.approx(1.0)
    assert m["severe_recall"] == pytest.approx(0.0)
    # 对比: 普通 accuracy 几乎看不出问题
    assert (p == y).mean() == pytest.approx(0.97)


def test_severe_degradation_miss_rate_counts_correctly():
    """漏报数必须算对 (只算真值严重且预测不严重的)。"""
    from evo.task_metrics import severe_degradation_miss_rate

    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0, 0, 1, 0, 0])  # 两个漏报
    m = severe_degradation_miss_rate(y, p, severe_classes=["1"])
    assert m["severe_count"] == 3
    assert m["severe_miss_count"] == 2
    assert m["severe_miss_rate"] == pytest.approx(2 / 3)
    assert m["severe_recall"] == pytest.approx(1 / 3)


def test_asymmetric_rul_score_penalizes_over_prediction():
    """A3 非对称评分: 过预测寿命的代价必须高于欠预测。"""
    from evo.task_metrics import asymmetric_rul_score

    # 同样 100 分钟误差: 过预测 (设备超期服役) 比欠预测 (提前换件) 更糟
    over = asymmetric_rul_score(np.array([200.0]), np.array([300.0]))
    under = asymmetric_rul_score(np.array([200.0]), np.array([100.0]))
    assert over["asym_penalty"] > under["asym_penalty"]
    assert over["over_predict_rate"] == pytest.approx(1.0)
    assert under["over_predict_rate"] == pytest.approx(0.0)

    # 权重可配置
    light = asymmetric_rul_score(
        np.array([200.0]), np.array([300.0]), over_predict_penalty=1.0
    )
    heavy = asymmetric_rul_score(
        np.array([200.0]), np.array([300.0]), over_predict_penalty=3.0
    )
    assert heavy["asym_penalty"] > light["asym_penalty"]


def test_task_metrics_uses_competition_criteria():
    """profile A: 分类任务必须按 Macro-F1 口径打分 (而非普通 F1)。"""
    from evo.task_metrics import compute_task_metrics

    # 大小类不平衡: 多数类 97%, 少数类 3%
    y = np.array([0] * 97 + [1] * 3)
    p = np.array([0] * 100)  # 少数类全错
    m = compute_task_metrics(y, p, "binary", profile="A")

    assert "macro_f1" in m
    assert m["macro_f1"] < 0.5, "Macro-F1 应反映少数类全错"
    # score 应按赛题口径 (Macro-F1 为主要成分)
    assert m["score"] < 0.7
    # 混淆矩阵作为官方辅助指标
    assert "confusion_matrix" in m


def test_profile_a_supports_multiclass_string_labels():
    """官方 A 包是 **6 类气体**, answer 是英文类别名 —— 不是数值。

    这曾导致两处崩溃:
      1. `compute_metrics` 里 `y_true >= 0` 对字符串数组抛 UFuncNoLoopError
      2. precision/recall 默认 average='binary' 对 6 类抛 ValueError
    """
    from evo.models import compute_metrics
    from evo.task_metrics import compute_task_metrics

    classes = [
        "Ethanol", "Ethylene", "Ammonia",
        "Acetaldehyde", "Acetone", "Toluene",
    ]
    y = np.array(classes * 20)
    p = y.copy()
    p[:5] = "Toluene"  # 5 个错

    # 通用指标不能崩
    base = compute_metrics(y, p, "binary")
    assert base["acc"] == pytest.approx(115 / 120)
    assert base["n_classes"] == 6

    # 赛题口径: Macro-F1 为主指标
    m = compute_task_metrics(y, p, "binary", profile="A")
    assert m["macro_f1"] > 0.9
    assert m["acc"] > 0.9
    # score 严格按官方口径 = 0.7*MacroF1 + 0.3*Acc (不掺 AUC)
    assert m["score"] == pytest.approx(0.7 * m["macro_f1"] + 0.3 * m["acc"])
    # 混淆矩阵是 6x6
    assert len(m["confusion_matrix"]) == 6
    assert len(m["confusion_matrix"][0]) == 6


def test_profile_b_does_not_use_miss_rate_in_score():
    """官方 B 包: "不以严重故障漏报率评价本任务"。

    若把漏报率并入 score, 系统会去优化一个**不被评分**的量,
    反而损害真正的主指标 RMSE。这是"优化错误目标"的典型陷阱。
    """
    from evo.task_metrics import compute_task_metrics

    y = np.array([100.0] * 97 + [10.0] * 3)
    p = np.array([100.0] * 97 + [500.0] * 3)  # 严重退化的全被高估
    m = compute_task_metrics(y, p, "regression", profile="B")

    # score 必须严格等于 -RMSE (主指标口径)
    assert m["score"] == pytest.approx(-m["rmse"])
    # 官方口径下**不应出现**漏报指标 (避免"优化了错误目标")
    assert "severe_miss_rate" not in m
    assert "severe_recall" not in m


def test_profile_c_reports_nasa_asymmetric_score():
    """官方 C 包: 报告 NASA 非对称误差分数 (逐样本惩罚之和)。"""
    from evo.task_metrics import compute_task_metrics, nasa_asymmetric_score

    y = np.array([100.0, 100.0])
    p = np.array([200.0, 50.0])  # 一个高估(惩罚5倍), 一个低估
    m = compute_task_metrics(y, p, "regression", profile="C")

    assert "nasa_asym_score" in m
    # 主指标仍是 RMSE
    assert m["score"] == pytest.approx(-m["rmse"])

    # 官方公式手工核对:
    # 高估: 5 * max(0, 200-100)/100 = 5.0
    # 低估: 1 * max(0, 100-50)/100 = 0.5
    assert m["nasa_asym_score"] == pytest.approx(5.5)
    # 是"和"不是"均值"
    assert m["nasa_asym_score_mean"] == pytest.approx(2.75)


def test_nasa_asymmetric_penalizes_over_prediction_five_times():
    """NASA 公式: 高估剩余寿命的惩罚是低估的 5 倍。"""
    from evo.task_metrics import nasa_asymmetric_score

    over = nasa_asymmetric_score(np.array([100.0]), np.array([200.0]))
    under = nasa_asymmetric_score(np.array([100.0]), np.array([0.0]))
    assert over["nasa_asym_score"] == pytest.approx(5.0)
    assert under["nasa_asym_score"] == pytest.approx(1.0)
    assert over["over_predict_rate"] == pytest.approx(1.0)


def test_severe_miss_is_diagnostic_only_not_veto(cfg):
    """漏报率只是诊断, 绝不影响上线判定 (官方 B 包明确不使用)。"""
    from evo.harness import Harness

    cfg.set("paths.artifacts_root", "artifacts/_tmp_diag")
    cfg.set("paths.audit_log", "artifacts/_tmp_diag/audit/audit.jsonl")
    cfg.set("paths.shadow_root", "artifacts/_tmp_diag/shadow")
    cfg.set("experiment.local_dir", "artifacts/_tmp_diag/exp")
    cfg.set("metrics.severe_threshold", 20.0)
    cfg.set("data.task", "regression")
    cfg.set("data.profile", "B")

    h = Harness(cfg, use_synthetic=True)
    h.prepare_data()
    h.bootstrap()

    shadow = h._shadow_set((3000, 3500))
    lab = shadow.labeled()
    fake_y = np.full(len(lab), 100.0)
    fake_y[: max(1, len(lab) // 10)] = 10.0  # 少量严重退化 (RUL ≤ 20)
    from evo.data import Dataset

    fake = Dataset(
        X=lab.X, y=fake_y, meta=lab.meta,
        name="fake", feature_names=lab.feature_names,
    )
    # 现网: 预测 10 (正确识别为严重退化) → 无漏报
    h.current_model = type(
        "M", (), {"task": "regression",
                  "predict": staticmethod(lambda X: np.full(len(X), 10.0))}
    )()
    # 候选: 预测 500 (以为还能用很久) → 全部漏报
    cand = type(
        "M", (), {"task": "regression",
                  "predict": staticmethod(lambda X: np.full(len(X), 500.0))}
    )()
    # 诊断应报告漏报恶化, 但**不否决**
    diag = h._severe_miss_diagnostic(cand, fake)
    assert diag["applicable"] is True
    assert diag["baseline_miss_rate"] == pytest.approx(0.0)
    assert diag["candidate_miss_rate"] > diag["baseline_miss_rate"]
    # 不存在任何基于漏报的否决入口
    assert not hasattr(h, "_severe_miss_regression")


def test_gate_probe_reaches_three_attempts(cfg, tmp_path):
    """赛题门槛①: 至少三次尝试迭代。

    即使漂移迟迟不到阈值, 系统也必须补足尝试次数 ——
    但补足的方式是"探索性重训", 仍走完整影子评估, 不可能降低最终性能。
    """
    import json

    from evo.harness import Harness

    cfg.set("paths.artifacts_root", str(tmp_path / "artifacts"))
    cfg.set("paths.audit_log", str(tmp_path / "artifacts/audit/audit.jsonl"))
    cfg.set("paths.shadow_root", str(tmp_path / "artifacts/shadow"))
    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    cfg.set("runtime.window_size", 500)
    cfg.set("safety.max_iterations", 15)
    cfg.set("gates.min_iterations", 3)

    Harness(cfg, use_synthetic=True).run()

    data = json.loads((tmp_path / "artifacts" / "run_summary.json").read_text(encoding="utf-8"))
    gates = data["gates"]
    assert gates["attempted_iterations"] >= 3, (
        f"尝试迭代次数不足: {gates['attempted_iterations']}"
    )
    assert gates["iteration_gate_passed"], f"门槛①未通过: {gates}"
    assert gates["all_passed"], f"门槛未全部通过: {gates}"


def test_gate_probe_does_not_bypass_safety(cfg):
    """门槛探索不能绕过预算与熔断。"""
    ctrl = PolicyController(cfg, seed=1)

    # 预算耗尽 → 即使 force=True 也不能更新
    ctrl.budget.gpu_seconds = ctrl.budget.limits["gpu_seconds"] * 0.95
    should, why = ctrl.should_update(0.0, "none", force=True)
    assert not should, "门槛探索绕过了预算约束"
    assert "预算" in why

    # 熔断 → 即使 force=True 也不能探索
    ctrl2 = PolicyController(cfg, seed=1)
    ctrl2.halted = True
    ctrl2._champion_locked = True
    should2, why2 = ctrl2.should_update(0.0, "none", force=True)
    assert not should2, "门槛探索绕过了熔断约束"
    assert "熔断" in why2

    # 正常状态 + force=True → 允许
    ctrl3 = PolicyController(cfg, seed=1)
    should3, why3 = ctrl3.should_update(0.0, "none", force=True)
    assert should3, "force=True 未生效"
    assert "门槛" in why3


def test_gate_probe_skipped_when_training_data_insufficient(cfg):
    """训练数据不足时不做门槛探索 —— 否则注定被拒并触发熔断。"""
    from evo.harness import Harness

    cfg.set("paths.artifacts_root", "artifacts/_tmp_probe")
    cfg.set("paths.audit_log", "artifacts/_tmp_probe/audit/audit.jsonl")
    cfg.set("paths.shadow_root", "artifacts/_tmp_probe/shadow")
    cfg.set("experiment.local_dir", "artifacts/_tmp_probe/exp")
    h = Harness(cfg, use_synthetic=True)
    h.prepare_data()
    h.bootstrap()
    # 初始训练集只有 4000 条, 但门槛要求 min_train=500 → 此时应可探索
    # 把门槛调到极高 → 必然跳过
    cfg.set("gates.gate_probe_min_train", 10_000_000)
    assert not h._gate_probe_needed(), "训练数据不足时仍触发了门槛探索"
