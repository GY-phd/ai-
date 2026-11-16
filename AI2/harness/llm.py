# -*- coding: utf-8 -*-
"""Planner driven by the competition-specified model (Qwen3.8-27B).

The harness never bundles weights or starts a service: it speaks the
OpenAI-compatible chat-completions protocol to whatever endpoint the evaluation
environment exposes (``HARNESS_LLM_BASE_URL``), and the model id is pinned in
the config (``Qwen3.8-27B``; a PyTRiO post-trained 4B/27B checkpoint is selected
by changing only that id).

Everything the model returns is treated as untrusted: the JSON is validated
against a strict schema, the action name must be in the allowed set, and any
parse/transport failure makes the caller fall back to the deterministic policy
(``harness.policy``).  That fallback is what lets the same image run offline.
"""
import json
import time


class LLMUnavailable(RuntimeError):
    pass


class LLMInvalid(ValueError):
    pass


ALLOWED_ACTIONS = ("do_nothing", "recalibrate_inputs", "retrain_head", "hpo", "rollback", "adapt")
RISKS = ("low", "medium", "high")

SYSTEM_PROMPT = (
    "You are the decision layer of an industrial model self-evolution harness. "
    "You receive a drift diagnosis card, the remaining budget, the deployed model's record and the last decisions. "
    "Choose exactly one action from: do_nothing, recalibrate_inputs, retrain_head, adapt, hpo, rollback. "
    "'adapt' retrains the candidate on the initial data plus the stream observed so far, which is the "
    "action that actually lets the deployed model follow a drifting population. "
    "Spending a training budget must be justified by the evidence; doing nothing is often the correct, "
    "defensible engineering decision. Reply with a single JSON object and nothing else."
)

USER_TEMPLATE = """DIAGNOSIS
{card}

DEPLOYED MODEL (public validation, minutes)
{champion}

REMAINING BUDGET
{budget}

RECENT DECISIONS
{history}

RULES
- retrain_head: cheap, refits only the output layer on the public replay buffer.
- recalibrate_inputs: unsupervised input statistics alignment, no labels used.
- hpo: trains new candidates on the public training set; most expensive.
- rollback: restore the previous champion when the current one regressed.
- If evidence is weak or the budget is nearly gone, choose do_nothing.

Return JSON with keys:
action (one of the five), params (object), hypothesis (<=200 chars), expected_gain_pct (number), risk (low|medium|high), reason (<=300 chars).
"""


class QwenPlanner:
    def __init__(self, settings, skills=None):
        self.model_id = settings["model_id"]
        self.base_url = (settings.get("base_url") or "").rstrip("/")
        self.api_key = settings.get("api_key") or ""
        self.timeout = float(settings.get("timeout_s", 30))
        self.max_tokens = int(settings.get("max_tokens", 1024))
        self.temperature = float(settings.get("temperature", 0.0))
        self.mode = settings.get("mode", "auto")
        self.skills = skills or ""
        self.tokens_used = 0
        self.calls = 0
        self.failures = 0
        self.last_error = None
        self._client = None

    # ------------------------------------------------------------------ status
    def available(self):
        if self.mode == "policy":
            return False
        if not self.base_url:
            return False
        try:
            return self._http_ready()
        except Exception as exc:  # noqa: BLE001 - availability must never raise
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def _http_ready(self):
        import urllib.request
        request = urllib.request.Request(self.base_url + "/models", headers=self._headers())
        with urllib.request.urlopen(request, timeout=3) as response:
            return 200 <= response.status < 300

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    # --------------------------------------------------------------- inference
    def plan(self, card, champion, budget, history):
        if not self.available():
            raise LLMUnavailable(self.last_error or "no endpoint configured")
        payload = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + ("\n\n" + self.skills if self.skills else "")},
                {"role": "user", "content": USER_TEMPLATE.format(
                    card=json.dumps(card, ensure_ascii=False, indent=2),
                    champion=json.dumps(champion, ensure_ascii=False, indent=2),
                    budget=json.dumps(budget, ensure_ascii=False, indent=2),
                    history=json.dumps(history[-4:], ensure_ascii=False, indent=2))},
            ],
        }
        started = time.monotonic()
        raw = self._post("/chat/completions", payload)
        self.calls += 1
        self.tokens_used += int(raw.get("usage", {}).get("total_tokens", 0))
        content = raw["choices"][0]["message"]["content"]
        plan = _extract_json(content)
        validated = validate_plan(plan)
        validated["_meta"] = {"planner": "qwen", "model_id": self.model_id,
                              "latency_s": round(time.monotonic() - started, 1),
                              "tokens": int(raw.get("usage", {}).get("total_tokens", 0))}
        return validated

    def _post(self, path, payload):
        import urllib.request
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode("utf-8"),
                                         headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise LLMUnavailable(self.last_error) from exc


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise LLMInvalid("model reply contains no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMInvalid(f"model reply is not valid JSON: {exc}") from exc


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise LLMInvalid("plan must be an object")
    action = plan.get("action")
    if action not in ALLOWED_ACTIONS:
        raise LLMInvalid(f"action {action!r} is not allowed")
    hypothesis = plan.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise LLMInvalid("hypothesis is required")
    params = plan.get("params", {})
    if not isinstance(params, dict):
        raise LLMInvalid("params must be an object")
    try:
        gain = float(plan.get("expected_gain_pct", 0.0))
    except (TypeError, ValueError) as exc:
        raise LLMInvalid("expected_gain_pct must be numeric") from exc
    risk = plan.get("risk", "medium")
    if risk not in RISKS:
        raise LLMInvalid(f"risk {risk!r} is not allowed")
    reason = plan.get("reason", "")
    if not isinstance(reason, str):
        raise LLMInvalid("reason must be a string")
    return {"action": action, "params": params, "hypothesis": hypothesis.strip()[:300],
            "expected_gain_pct": max(-100.0, min(100.0, gain)), "risk": risk,
            "reason": reason.strip()[:400], "_meta": {"planner": "qwen"}}
