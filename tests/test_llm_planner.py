# -*- coding: utf-8 -*-
"""Planner transport: thinking-trace fallback, retries, cache, prompt budget.

These tests use a fake ``_post`` so they never touch the network.  They pin
the four behaviours that turn a 27B reasoning model from "288 s and an empty
reply" into a usable 4-core decision layer.
"""
import json
import unittest

from harness.llm import (
    LLMUnavailable,
    QwenPlanner,
    _extract_json,
    compact_history,
)

PLAN = json.dumps({
    "action": "do_nothing",
    "params": {},
    "hypothesis": "no actionable evidence",
    "expected_gain_pct": 0.0,
    "risk": "low",
    "reason": "keep the champion",
})

CARD = {"drift_type": "none", "confidence": 0.8, "needs_labels": False}
CHAMPION = {"rmse": 500.0, "mae": 400.0}
BUDGET = {"experiments_left": 3, "train_seconds_left": 600, "tokens_left": 1000,
          "wall_seconds_left": 900}
HISTORY = [
    {"cycle": 1, "action": "hpo", "decision": "reject", "hypothesis": "x" * 300,
     "reason": "y" * 300, "gate": {"decision": "reject", "checks": {"gain_pct": -1.5}},
     "diagnosis": {"summary": "large field that should not enter the prompt"},
     "drift": {"features": ["large field"]}, "cost_seconds": 12.0},
]


def reply(content, tokens=10):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": tokens}}


class FakePlanner(QwenPlanner):
    def __init__(self, replies):
        super().__init__({"model_id": "Qwen3.8-27B", "base_url": "http://example/v1",
                          "mode": "auto", "max_tokens": 1024, "temperature": 0.0,
                          "retries": 3, "backoff_s": 0.0, "cache": True})
        self.replies = list(replies)
        self.payloads = []
        self.calls = 0

    def available(self):
        return True

    def _post(self, path, payload):
        self.payloads.append(payload)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class PlannerTest(unittest.TestCase):
    def test_trace_is_disabled_on_the_first_attempt(self):
        planner = FakePlanner([reply(PLAN)])
        planner.plan(CARD, CHAMPION, BUDGET, HISTORY)
        self.assertIs(planner.payloads[0]["enable_thinking"], False)

    def test_a_host_that_rejects_the_option_is_asked_again_without_it(self):
        planner = FakePlanner([LLMUnavailable("unknown option"), reply(PLAN)])
        plan = planner.plan(CARD, CHAMPION, BUDGET, HISTORY)
        self.assertEqual(plan["action"], "do_nothing")
        self.assertEqual(plan["_meta"]["attempts"], 2)
        self.assertEqual(planner.calls, 1)
        self.assertNotIn("enable_thinking", planner.payloads[1])

    def test_empty_content_is_retried(self):
        planner = FakePlanner([reply("", tokens=100), reply(PLAN)])
        plan = planner.plan(CARD, CHAMPION, BUDGET, HISTORY)
        self.assertEqual(plan["action"], "do_nothing")
        self.assertEqual(planner.calls, 2)
        self.assertEqual(planner.failures, 1)

    def test_identical_states_reuse_the_cached_plan(self):
        planner = FakePlanner([reply(PLAN)])
        first = planner.plan(CARD, CHAMPION, BUDGET, HISTORY)
        second = planner.plan(CARD, CHAMPION, BUDGET, HISTORY)
        self.assertEqual(planner.calls, 1)
        self.assertFalse(first["_meta"]["cache_hit"])
        self.assertTrue(second["_meta"]["cache_hit"])
        self.assertEqual(second["_meta"]["tokens"], 0)

    def test_exhausted_token_budget_never_calls_the_model(self):
        planner = FakePlanner([reply(PLAN)])
        with self.assertRaises(LLMUnavailable):
            planner.plan(CARD, CHAMPION, {**BUDGET, "tokens_left": 0}, HISTORY)
        self.assertEqual(planner.calls, 0)

    def test_compact_history_drops_large_fields_and_keeps_policy_inputs(self):
        row = compact_history(HISTORY)[0]
        self.assertEqual(row["action"], "hpo")
        self.assertEqual(row["decision"], "reject")
        self.assertNotIn("diagnosis", row)
        self.assertNotIn("drift", row)
        self.assertLessEqual(len(row["hypothesis"]), 160)

    def test_think_blocks_are_stripped_before_parsing(self):
        parsed = _extract_json("<think>{not json}</think>\n" + PLAN)
        self.assertEqual(parsed["action"], "do_nothing")


if __name__ == "__main__":
    unittest.main()
