# -*- coding: utf-8 -*-
"""Planner driven by the competition-specified model (Qwen3.8-27B).

The harness never bundles weights or starts a service: it speaks the
OpenAI-compatible chat-completions protocol to whatever endpoint the evaluation
environment exposes (``HARNESS_LLM_BASE_URL``), and the model id is pinned in
the config (``Qwen3.8-27B``; a PyTRiO post-trained 4B/27B checkpoint is selected
by changing only that id).

    10|Everything the model returns is treated as untrusted: the JSON is validated
against a strict schema, the action name must be in the allowed set, and any
parse/transport failure makes the caller fall back to the deterministic policy
(``harness.policy``).  That fallback is what lets the same image run offline.

The model id and the platform's id
----------------------------------
The platform's dropdown lists the models under their **PyTRiO** ids, which
use a slash::

    Qwen/Qwen3.8-27B
    Qwen/Qwen3.5-4B

while the short form used in this project's configuration and in the
competition's own text uses a dot::

    Qwen3.8-27B

An endpoint that speaks the PyTRiO protocol will reject the short form,
and an endpoint that expects the short form will reject the slashed one.
So the planner resolves the id against the two known aliases before it
speaks: ``MODEL_ALIASES`` maps the short form onto the platform's id,
and a model that is already slashed is left alone.  The alias table is
the single place where the two spellings meet, so a future model only
has to be added once.

This module is the **single authority** on model identity
----------------------------------------------------------
There are two questions about a model id, and they are different
questions:

**Which spelling?**  ``resolve_model_id`` answers that: it turns the
competition's short form into the platform's spelling.  It is a pure
spelling question and it never decides whether a model may be used.

**Is it allowed?**  ``is_allowed_model`` answers that: it compares an id
against the competition's allow-list and the platform's post-training
derivatives.  It never rewrites the id.

The split matters.  Rewriting an unknown id into an allowed one would
hide a misconfiguration behind a tidy request, and then the call would
carry a model the operator never chose.  So the two never share a
function, and neither one does the other's job.

``normalize_model_id`` is the shared rule underneath both: it strips
every separator and folds the case, so ``Qwen3.8-27B``,
``Qwen3-8-27B``, ``qwen3_8_27b`` and ``Qwen/Qwen3.8-27B`` all reduce
to the same text.  That is what makes a post-training derivative such
as ``qwen3.8-27b-ft-20261115`` recognisable as a derivative rather
than as an unknown model.
"""
import json
import time


#: The platform's model ids, and the short forms the competition's text
#: uses for them.  The platform's spelling is the one that is sent.
MODEL_ALIASES = {
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-4b-instruct": "Qwen/Qwen3.5-4B",
}


def normalize_model_id(model_id):
    """Fold an id to its comparison form.

    The rule is deliberately blunt: every separator is dropped and the
    case is folded.  Two ids that differ only in spelling - a dot for a
    dash, a slash for a dash, an upper case letter for a lower case
    one - compare equal, because the platform's own naming has moved
    between those spellings before, and the competition's allow-list
    is written in the short form while the platform serves the long one.

    A **vendor prefix** is dropped before the folding.  The platform
    serves ``Qwen/Qwen3.8-27B`` while the competition writes
    ``Qwen3.8-27B``, so the two spellings of one model differ by a
    segment as well as by punctuation.  Without dropping it, the
    platform's own spelling would not match the allow-list entry it is
    supposed to match.

    An empty id normalises to an empty string, so a caller can compare
    it without a special case.
    """
    text = str(model_id or "").strip()
    if "/" in text:
        # The platform's form is ``vendor/model``: the vendor segment is
        # a namespace, not part of the model's own name.
        text = text.rsplit("/", 1)[-1]
    return "".join(ch for ch in text.lower() if ch.isalnum())


def resolve_model_id(model_id):
    """Map a model id onto the spelling the endpoint expects.

    An id that already carries the platform's slash is returned
    unchanged; a short form is mapped through ``MODEL_ALIASES``.  An
    unknown id is returned as it came in - the whitelist check in
    ``evo`` decides whether it may be used at all, and this function
    only answers "which spelling".
    """
    text = str(model_id or "").strip()
    if not text:
        return text
    if "/" in text:
        return text
    return MODEL_ALIASES.get(text.lower(), text)


def is_allowed_model(model_id, allowed, derived=()):
    """Is this model id one the competition permits?

    The judgement has three layers, in this order:

    1. **An explicit exemption.**  An id the configuration names as a
       platform post-training derivative is allowed outright.  This
       layer comes first because a derivative's name is rewritten by
       the platform, and a literal comparison would call a compliant
       model a violation.

    2. **A compliant base.**  The normalised id *contains* one of the
       allowed bases.  This is what lets ``qwen3.8-27b-ft-20261115``
       through on the strength of its ``qwen3.8-27b`` base without
       maintaining a long list of suffixes.

    3. **Otherwise, refused.**  A model outside the allow-list is not
       permitted, and the caller must not issue the request.

    The containment in layer 2 is a **substring** test on the
    normalised text, so a separator variant of an allowed base is
    accepted as well.
    """
    normalised = normalize_model_id(model_id)
    if not normalised:
        return False

    # 1) The platform's post-training derivatives (the second route).
    for name in derived or ():
        candidate = normalize_model_id(name)
        if candidate and candidate in normalised:
            return True

    # 2) A compliant base.
    return any(normalize_model_id(name) in normalised
               for name in allowed or () if normalize_model_id(name))


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
        # The config carries the short form; the endpoint expects the
        # platform's spelling.  The resolution happens once, here, so
        # every request carries the same id.
        self.model_id = resolve_model_id(settings["model_id"])
        self.configured_model_id = str(settings["model_id"])
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
