"""SwanLab 记录链路测试 —— 用 mock 验证, 不需要网络与 API Key。

覆盖的真实风险:
  1. `swanlab.init()` 的参数名在新旧版本不同 (name/experiment_name,
     log_dir/logdir) —— 传错会 TypeError, 实验根本建不起来。
  2. `swanlab.log()` 接收**扁平指标字典** —— 传嵌套结构会得到一个
     叫 "metrics" 的怪指标, 网页上什么都看不到。
  3. 没有 `init()` 就调 `log()` —— SwanLab 收不到任何数据。
     (早期实现从未调用 start_run, 导致 SwanLab 上是空的。)
  4. step 参数 —— 不用它, 所有点挤在同一 step, 曲线退化成水平线。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _MockSwan:
    """最小 SwanLab mock —— 记录 init/log/finish 的调用。"""

    def __init__(self, accept_public: bool = True, supports_step: bool = True):
        self.accept_public = accept_public
        self.supports_step = supports_step
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, object]] = []
        self.finish_calls = 0
        self.login_calls: list[dict] = []

    def init(self, **kwargs):
        # 模拟新版签名校验: 只接受确实存在的参数
        allowed = {
            "workspace", "project", "public", "name", "color", "description",
            "job_type", "group", "tags", "id", "resume", "parallel", "config",
            "settings", "callbacks", "reinit", "log_dir", "mode",
            "on_online_error", "experiment_name", "logdir",
        }
        bad = set(kwargs) - allowed
        if bad:
            raise TypeError(f"unexpected keyword argument: {sorted(bad)}")
        if self.accept_public and "public" not in kwargs:
            # 模拟"必须显式声明公开性"的版本
            pass
        self.init_calls.append(kwargs)
        return {"run": "mock"}

    def log(self, data, step=None):
        if step is not None and not self.supports_step:
            raise TypeError("log() got an unexpected keyword argument 'step'")
        self.log_calls.append((dict(data), step))

    def finish(self):
        self.finish_calls += 1

    def login(self, api_key=None, relogin=False, host=None, save=False, timeout=10):
        self.login_calls.append({"api_key": bool(api_key), "save": save})
        return True


@pytest.fixture
def swan_cfg(tmp_path, monkeypatch):
    """构造一个启用 swanlab 的配置 (offline=False)。"""
    from evo.config import load_config

    cfg = load_config()
    cfg.set("runtime.offline", False)
    cfg.set("experiment.local_dir", str(tmp_path / "exp"))
    cfg.set("experiment.project", "zxyq11")
    cfg.set("experiment.workspace", "GY-NJph")
    return cfg


def _patch_swan(monkeypatch, mock):
    import sys
    import types

    module = types.ModuleType("swanlab")
    module.init = mock.init
    module.log = mock.log
    module.finish = mock.finish
    module.login = mock.login
    module.__version__ = "0.10.4-mock"
    monkeypatch.setitem(sys.modules, "swanlab", module)


def test_swanlab_init_uses_correct_parameter_names(swan_cfg, monkeypatch):
    """init() 必须用新版的 `name` / `log_dir` (旧名会 TypeError)。"""
    from evo.experiment import ExperimentRecorder

    mock = _MockSwan()
    _patch_swan(monkeypatch, mock)

    rec = ExperimentRecorder(swan_cfg)
    rec.start_run("evo_test", config={"seed": 1}, tags=["t"])

    assert mock.init_calls, "init 未被调用"
    call = mock.init_calls[0]
    assert call["name"] == "evo_test"
    assert "log_dir" in call
    assert call["project"] == "zxyq11"
    assert call["workspace"] == "GY-NJph"
    assert call["config"] == {"seed": 1}
    # 公开性必须显式声明 (赛题硬要求)
    assert call.get("public") is True
    assert rec._swan_public is True


def test_swanlab_log_sends_flat_metrics_with_step(swan_cfg, monkeypatch):
    """log() 必须传**扁平**指标 + step, 否则网页上看不出曲线。"""
    from evo.experiment import ExperimentRecord, ExperimentRecorder

    mock = _MockSwan()
    _patch_swan(monkeypatch, mock)

    rec = ExperimentRecorder(swan_cfg)
    rec.start_run("evo_test", config={})
    rec.log(
        ExperimentRecord(
            name="iteration_001/retrain",
            config={"model": "lgbm"},
            metrics={"score": -123.5, "rmse": 123.5},
            tags=["iteration"],
            notes="测试",
            stage="retrain",
            iteration=3,
            cost_gpu_seconds=1.25,
            cost_labels=250,
        )
    )

    assert mock.log_calls, "log 未被调用"
    data, step = mock.log_calls[0]
    # 扁平: 不能有嵌套的 metrics/config/tags/notes
    assert "metrics" not in data
    assert "config" not in data
    assert "notes" not in data
    # 指标被展平
    assert data["score"] == pytest.approx(-123.5)
    assert data["rmse"] == pytest.approx(123.5)
    assert data["cost_gpu_seconds"] == pytest.approx(1.25)
    assert data["cost_labels"] == pytest.approx(250.0)
    # step 用迭代序号
    assert step == 3


def test_swanlab_log_works_without_step_support(swan_cfg, monkeypatch):
    """旧版 log() 不支持 step —— 必须自动降级而不是崩。"""
    from evo.experiment import ExperimentRecord, ExperimentRecorder

    mock = _MockSwan(supports_step=False)
    _patch_swan(monkeypatch, mock)

    rec = ExperimentRecorder(swan_cfg)
    rec.start_run("evo_test", config={})
    rec.log(
        ExperimentRecord(
            name="r", metrics={"x": 1.0}, stage="iteration", iteration=2
        )
    )
    assert mock.log_calls, "降级路径也没写入"


def test_swanlab_public_not_confirmed_when_unsupported(swan_cfg, monkeypatch):
    """版本不接受任何公开参数时 —— 必须**诚实标记为未确认**。

    绝不能谎报公开性, 否则 compliance 报告会给出错误的合规结论
    (赛题硬要求: 记录必须公开)。
    """
    from evo.experiment import ExperimentRecorder

    class _NoPublic(_MockSwan):
        def init(self, **kwargs):
            if "public" in kwargs or "visibility" in kwargs or "mode" in kwargs:
                raise TypeError("no public param in this version")
            return super().init(**kwargs)

    mock = _NoPublic()
    _patch_swan(monkeypatch, mock)

    rec = ExperimentRecorder(swan_cfg)
    rec.start_run("evo_test", config={})
    assert rec._swan_public is False
    assert rec._public_set_failed is True
    report = rec.compliance_report()
    assert report["swanlab_public_ok"] is False


def test_swanlab_falls_back_to_local_when_not_logged_in(swan_cfg, monkeypatch):
    """未登录时必须降级到本地记录, 不得让闭环失败。"""
    from evo.experiment import ExperimentRecorder

    class _LoginFail(_MockSwan):
        def login(self, **kwargs):
            raise RuntimeError("no api key")

    mock = _LoginFail()
    _patch_swan(monkeypatch, mock)

    rec = ExperimentRecorder(swan_cfg)
    rec.start_run("evo_test", config={})
    assert rec.mode == "local"
    # 本地记录仍然可用
    rec.log(
        __import__("evo.experiment", fromlist=["ExperimentRecord"]).ExperimentRecord(
            name="r", metrics={"x": 1.0}
        )
    )
    assert len(rec.local.load()) == 1


def test_swanlab_login_prefers_env_api_key(swan_cfg, monkeypatch):
    """环境变量 SWANLAB_API_KEY 优先 (不落盘、更适合 CI)。"""
    from evo.experiment import ExperimentRecorder

    mock = _MockSwan()
    _patch_swan(monkeypatch, mock)
    monkeypatch.setenv("SWANLAB_API_KEY", "test-key-123")

    rec = ExperimentRecorder(swan_cfg)
    assert rec._ensure_login() is True
    assert mock.login_calls
    assert mock.login_calls[0]["api_key"] is True
    assert mock.login_calls[0]["save"] is True


def test_harness_calls_start_run(swan_cfg, monkeypatch):
    """Harness 必须调用 start_run —— 否则 SwanLab 上一个实验都没有。

    这曾是一个真实缺陷: 只有 log() 没有 init(), SwanLab 收不到数据。
    """
    from evo.harness import Harness

    mock = _MockSwan()
    _patch_swan(monkeypatch, mock)

    swan_cfg.set("runtime.window_size", 400)
    swan_cfg.set("safety.max_iterations", 3)
    swan_cfg.set("paths.artifacts_root", str(Path(swan_cfg.path("experiment.local_dir")).parent / "art"))
    swan_cfg.set("paths.audit_log", "art/audit/audit.jsonl")
    swan_cfg.set("paths.shadow_root", "art/shadow")

    h = Harness(swan_cfg, use_synthetic=True)
    h.run()

    assert mock.init_calls, "Harness 未调用 init()"
    call = mock.init_calls[0]
    # 实验名必须唯一且可读
    assert call["name"].startswith("evo_")
    # config 必须包含可对比的超参
    cfg = call["config"]
    for key in ("seed", "task", "profile", "window_size", "base_model"):
        assert key in cfg, f"config 缺少 {key}"
    # 必须有 log 与 finish
    assert mock.log_calls
    assert mock.finish_calls >= 1


def test_stage_id_is_stable_across_processes():
    """阶段 id 必须跨进程稳定 —— 内置 hash() 每进程随机, 会让数据对不上。"""
    from evo.experiment import _stage_id

    # 已知阶段走固定表
    assert _stage_id("bootstrap") == 0.0
    assert _stage_id("summary") == 5.0
    # 未知阶段: 同一输入必须同一输出
    assert _stage_id("custom_xyz") == _stage_id("custom_xyz")
