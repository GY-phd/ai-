# -*- coding: utf-8 -*-
"""Evaluate a planner checkpoint on held-out diagnosis cards.

The metric that matters for the harness is not free-form text quality: it is
whether the reply is a legal plan and whether its action matches the
deterministic policy on the same card.  The gate and the policy fallback still
protect the loop, but a decision layer that agrees with the policy can replace
the 27B at a fraction of the token cost.

Usage:
    python -m train.eval_planner --data train/data/planner-val.jsonl --limit 20
    python -m train.eval_planner --data train/data/planner-val.jsonl --limit 20 \
        --model-path trio://.../sampler_weights/planner-27b-sft-v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen3.8-27B"


def load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def chat_prompt(tokenizer, row):
    messages = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def evaluate(rows, base_model, model_path, max_tokens, name):
    import pytrio as trio
    from harness.llm import _extract_json, validate_plan

    service = trio.ServiceClient()
    client = (service.create_sampling_client(base_model=base_model, model_path=model_path)
              if model_path else service.create_sampling_client(base_model=base_model))
    tokenizer = client.get_tokenizer()
    params = trio.SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=42)

    records = []
    for row in rows:
        expected = json.loads(row["assistant"])
        tokens = chat_prompt(tokenizer, row)
        started = time.monotonic()
        result = client.sample(prompt=trio.ModelInput.from_ints(tokens), num_samples=1,
                               sampling_params=params).result()
        latency = time.monotonic() - started
        text = result.sequences[0].text if result.sequences else ""
        parse_ok = schema_ok = action_match = False
        action = error = None
        try:
            parsed = _extract_json(text)
            parse_ok = True
            plan = validate_plan(parsed)
            schema_ok = True
            action = plan["action"]
            action_match = action == expected["action"]
        except Exception as exc:  # noqa: BLE001 - a bad reply is a metric, not a crash
            error = f"{type(exc).__name__}: {exc}"
        records.append({
            "id": row.get("id"), "source": row.get("source"),
            "expected_action": expected["action"], "action": action,
            "parse_ok": parse_ok, "schema_ok": schema_ok, "action_match": action_match,
            "latency_s": round(latency, 1), "chars": len(text), "error": error,
            "reply": text[:300],
        })
        print(json.dumps({key: records[-1][key] for key in
                          ("id", "expected_action", "action", "schema_ok", "action_match", "latency_s")},
                         ensure_ascii=False), flush=True)

    total = len(records)
    actions = {}
    for record in records:
        key = record["action"] or "invalid"
        actions[key] = actions.get(key, 0) + 1
    summary = {
        "name": name, "model_path": model_path, "rows": total,
        "parse_rate": round(sum(r["parse_ok"] for r in records) / total, 3) if total else 0.0,
        "schema_rate": round(sum(r["schema_ok"] for r in records) / total, 3) if total else 0.0,
        "action_match_rate": round(sum(r["action_match"] for r in records) / total, 3) if total else 0.0,
        "mean_latency_s": round(sum(r["latency_s"] for r in records) / total, 1) if total else 0.0,
        "mean_chars": round(sum(r["chars"] for r in records) / total, 1) if total else 0.0,
        "actions": actions, "records": records,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="train/data/planner-val.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-path", default=None, help="save_weights_for_sampler() path")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default="train/reports")
    args = parser.parse_args()

    name = args.name or ("sft" if args.model_path else "base")
    rows = load_rows(args.data, args.limit)
    summary = evaluate(rows, args.base_model, args.model_path, args.max_tokens, name)
    out = Path(args.out) / f"eval-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("name", "model_path", "rows", "parse_rate", "schema_rate",
                       "action_match_rate", "mean_latency_s", "mean_chars", "actions")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
