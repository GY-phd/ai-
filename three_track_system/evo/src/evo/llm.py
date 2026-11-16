"""大模型接入层 (Harness 的"大脑")。

赛题要求: 驱动 Harness 的大模型须为
  * 选手通过 PyTRiO 平台训练得到的模型, 或
  * PyTRiO 的 Qwen3.8-27B 模型

因此本模块走 OpenAI 兼容接口 (PyTRiO 训推服务), 并提供:
  * 严格的重试/超时/Token 计量 (Token 是赛题客观分维度)
  * 无服务可用时的确定性规则降级, 保证离线评测不中断
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config


@dataclass
class LLMUsage:
    """Token 用量统计 — 直接对应赛题客观分的 Token 维度。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


class LLMClient:
    """PyTRiO 训推服务客户端 (OpenAI 兼容)。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = os.environ.get(
            str(cfg.get("llm.base_url_env", "PYTRIO_BASE_URL")), ""
        ).rstrip("/")
        self.api_key = os.environ.get(str(cfg.get("llm.api_key_env", "PYTRIO_API_KEY")), "")
        self.model = os.environ.get(
            str(cfg.get("llm.model_env", "PYTRIO_MODEL")), ""
        ) or str(cfg.get("llm.default_model", "Qwen3.8-27B"))
        self.temperature = float(cfg.get("llm.temperature", 0.2))
        self.max_tokens = int(cfg.get("llm.max_tokens", 1024))
        self.timeout = int(cfg.get("llm.timeout", 60))
        self.usage = LLMUsage()
        self._client = None
        # 属性名与服务可用性方法不能同名 —— 这里用 _service_ok 存储状态
        self._service_ok = bool(self.base_url)

        # -- 模型白名单校验 --
        # 赛题明确: "平台仅提供千问3.8 27B和千问3.5 4B两个模型可选,
        # 若想挑战可使用4B模型", 即**不允许**使用其他模型。
        # 这里做校验并留痕, 避免误配外部模型导致违规。
        #
        # **路线二 (PyTRiO 后训练) 的关键处理**:
        # 赛题 PPT 第 07 页给出两条路线:
        #   路线一 | 使用指定基模: Qwen3.8-27B 直接作为 Harness 的驱动模型
        #   路线二 | PyTRiO 后训练: 以 Qwen3.8-27B 或 4B 为**基座**,
        #            通过 PyTRiO 后训练获得驱动模型
        # 路线二产出的模型名通常带后缀 (如 "qwen3.8-27b-ft-20261115"),
        # 若只做字面相等校验, 会把**合规的后训练模型误判为违规**。
        # 因此这里分两层:
        #   1) 基座合规: 模型名里能识别出白名单基座 (归一化后包含)
        #   2) 显式豁免: 配置里可声明"平台后训练派生模型", 直接放行
        self.allowed_models = [
            str(m) for m in (cfg.get("llm.allowed_models") or [])
        ] or ["Qwen3.8-27B", "Qwen3.5-4B"]
        # 平台后训练派生模型名单 (路线二) —— 如 ["qwen3.8-27b-ft", ...]
        self.derived_models = [
            str(m) for m in (cfg.get("llm.derived_models") or [])
        ]
        self.model_allowed = self._check_model_allowed(self.model)

        if self._service_ok:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "EMPTY",
                    timeout=self.timeout,
                )
            except Exception:
                try:
                    import requests

                    self._session = requests.Session()
                    self._client = "requests"
                except Exception:
                    self._client = None
                    self._service_ok = False

    # -- 调用 ------------------------------------------------------------
    def _check_model_allowed(self, model: str) -> bool:
        """校验模型是否在赛题允许的白名单内。

        三层判定 (按顺序):
          1) **显式豁免**: 命中 `llm.derived_models` (平台后训练派生模型, 路线二)
             → 直接允许。这层放在最前, 因为后训练模型名通常带后缀,
             字面匹配会误判违规。
          2) **基座合规**: 归一化后模型名**包含**白名单基座
             (如 "qwen3.8-27b-ft" 包含 "qwen3827b") → 允许。
             这样路线二的命名变体无需维护长名单也能通过。
          3) 否则拒绝 —— 且**不会真正发起调用**, 避免违规。

        归一化规则: 忽略大小写与分隔符 —— 平台上的实际模型名可能是
        "qwen3.8-27b" / "Qwen3-8-27B" / "qwen3_8_27b" 等变体,
        直接字符串相等会误判。
        """
        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
        n = norm(model)

        # 1) 平台后训练派生模型 (路线二) —— 显式豁免
        for d in self.derived_models:
            if norm(d) and norm(d) in n:
                return True

        # 2) 基座合规
        return any(norm(m) in n for m in self.allowed_models)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int | None = None) -> str:
        """调用大模型; 失败时返回空串并计数 (由上层触发降级)。"""
        if not self._service_ok or self._client is None:
            return ""
        # 白名单校验失败 → 不调用 (违规风险), 走规则降级
        if not self.model_allowed:
            return ""
        mt = max_tokens or self.max_tokens
        try:
            if self._client == "requests":
                import requests

                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": mt,
                    },
                    headers={"Authorization": f"Bearer {self.api_key or 'EMPTY'}"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            else:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=mt,
                )
                data = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp)
            usage = data.get("usage") or {}
            self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.usage.completion_tokens += int(usage.get("completion_tokens", 0))
            self.usage.calls += 1
            return data["choices"][0]["message"]["content"]
        except Exception:
            self.usage.calls += 1
            return ""

    def available(self) -> bool:
        """大模型服务是否可用 (方法名与属性冲突风险, 故状态存于 _service_ok)。"""
        return self._service_ok and self._client is not None


# =====================================================================
# Harness 决策提示模板
# =====================================================================

DECISION_SYSTEM_PROMPT = """你是工业模型运维决策智能体 (Harness 的大脑)。
你的输入是漂移监测报告、归因结论与当前模型性能快照。
你必须输出严格的 JSON, 不要包含任何解释性文字或 Markdown 代码块。

输出格式:
{
  "action": "retrain" | "retrain_finetune" | "recalibrate" | "monitor" | "rollback",
  "target_model": "lgbm" | "logreg" | "rf" | "mlp",
  "reasoning": "不超过 120 字的中文理由",
  "expected_gain": 0.0,
  "risk_level": "low" | "medium" | "high",
  "needs_more_labels": true | false
}

约束:
- 若归因为 concept 漂移且标签覆盖不足, 必须设 needs_more_labels=true 且 action=monitor 或 retrain_finetune
- 若漂移严重度为 none, action 必须为 monitor
- expected_gain 必须落在 [0, 1]
- 禁止输出 train/test 之外的操作
"""

DECISION_USER_TEMPLATE = """【漂移监测报告】
{drift_report}

【归因结论】
{attribution}

【当前模型性能】
{performance}

【资源预算】
{budget}

请给出下一步决策的 JSON。
"""


# =====================================================================
# 规则降级 (确定性, 无 Token 开销)
# =====================================================================

def rules_decision(
    drift_type: str,
    severity: str,
    error_increase: float,
    label_coverage: float,
    current_model: str,
) -> Dict[str, Any]:
    """确定性决策 — LLM 不可用时的兜底, 也作为 LLM 输出的对照基线。"""
    if severity == "none":
        return {
            "action": "monitor",
            "target_model": current_model,
            "reasoning": "无显著漂移, 保持监控。",
            "expected_gain": 0.0,
            "risk_level": "low",
            "needs_more_labels": False,
        }

    if drift_type == "concept":
        if label_coverage < 0.5:
            return {
                "action": "monitor",
                "target_model": current_model,
                "reasoning": "概念漂移但标签覆盖不足, 先请求标注再重训。",
                "expected_gain": 0.0,
                "risk_level": "medium",
                "needs_more_labels": True,
            }
        return {
            "action": "retrain",
            "target_model": current_model,
            "reasoning": "概念漂移且标签充足, 重训以更新输入-输出关系。",
            "expected_gain": max(0.0, error_increase),
            "risk_level": "medium",
            "needs_more_labels": False,
        }

    if drift_type == "covariate_weak":
        return {
            "action": "monitor",
            "target_model": current_model,
            "reasoning": "轻度分布偏移, 误差未恶化, 观察一个窗口避免无谓更新成本。",
            "expected_gain": 0.0,
            "risk_level": "low",
            "needs_more_labels": False,
        }

    if drift_type in ("covariate", "mixed"):
        return {
            "action": "retrain",
            "target_model": current_model,
            "reasoning": f"{drift_type} 漂移, 重训以适配新分布。",
            "expected_gain": max(0.005, error_increase),
            "risk_level": "low",
            "needs_more_labels": False,
        }

    return {
        "action": "monitor",
        "target_model": current_model,
        "reasoning": "证据不足, 继续观察。",
        "expected_gain": 0.0,
        "risk_level": "low",
        "needs_more_labels": False,
    }


def parse_decision(raw: str) -> Optional[Dict[str, Any]]:
    """从 LLM 原始输出中稳健地解析 JSON。"""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    allowed_actions = {"retrain", "retrain_finetune", "recalibrate", "monitor", "rollback"}
    if obj.get("action") not in allowed_actions:
        return None
    try:
        obj["expected_gain"] = float(obj.get("expected_gain", 0.0))
    except (TypeError, ValueError):
        obj["expected_gain"] = 0.0
    obj["needs_more_labels"] = bool(obj.get("needs_more_labels", False))
    return obj


def build_decision_prompt(
    drift_report: Dict[str, Any],
    attribution: Dict[str, Any],
    performance: Dict[str, Any],
    budget: Dict[str, Any],
) -> List[Dict[str, str]]:
    """构造决策提示。"""
    user = DECISION_USER_TEMPLATE.format(
        drift_report=json.dumps(drift_report, ensure_ascii=False, indent=2),
        attribution=json.dumps(attribution, ensure_ascii=False, indent=2),
        performance=json.dumps(performance, ensure_ascii=False, indent=2),
        budget=json.dumps(budget, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
