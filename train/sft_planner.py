"""LoRA SFT of the harness experiment planner on PyTRIO.

The model learns one narrow behavior: given the same system prompt and JSON
context the harness sends at runtime, emit a JSON object with an ``analysis``
string and 1-3 executable trials. Prompt tokens are masked; only the assistant
JSON (plus the chat end token) contributes to the loss.

Examples:
    # show platform models after `trio login -k ...`
    python -m train.sft_planner --list-models

    # cheap smoke test
    python -m train.sft_planner --data train/data/planner-train.jsonl \
        --limit 4 --epochs 1 --batch-size 2 --save-name planner-smoke

    # real run
    python -m train.sft_planner --data train/data/planner-train.jsonl \
        --epochs 3 --batch-size 4 --lr 1e-4 --rank 32 \
        --save-name planner-sft-v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

DEFAULT_BASE_MODEL = os.environ.get("PYTRIO_BASE_MODEL", "Qwen/Qwen3.8-27B")


def _load_rows(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("system", "user", "assistant"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}: row {row.get('id')} missing {key}")
            json.loads(row["assistant"])  # fail fast on malformed labels
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def _apply_chat_template(tokenizer, messages: list[dict], **kwargs) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )


def _end_token(tokenizer) -> int | None:
    for token in ("<|im_end|>", "<|endoftext|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            return token_id
    return tokenizer.eos_token_id


def render(row: dict, tokenizer, max_seq_len: int):
    """Return a ``trio.Datum`` or ``(None, reason)`` when the row is too long."""
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]
    prompt = _apply_chat_template(tokenizer, messages, enable_thinking=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    completion_tokens = tokenizer.encode(row["assistant"], add_special_tokens=False)
    end = _end_token(tokenizer)
    if end is not None:
        completion_tokens = completion_tokens + [end]

    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    if len(tokens) < 2:
        return None, "too_short"
    if len(tokens) > max_seq_len:
        return None, f"too_long:{len(tokens)}"

    import pytrio as trio

    return (
        trio.Datum(
            model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": np.asarray(tokens[1:], dtype=np.int32),
                "weights": np.asarray(weights[1:], dtype=np.float32),
            },
        ),
        None,
    )


def _per_token_loss(result, datums) -> float:
    metrics = getattr(result, "metrics", None) or {}
    try:
        loss_sum = float(metrics["loss_sum"])
        tokens = float(metrics["token_count"])
        if tokens > 0:
            return loss_sum / tokens
    except (KeyError, TypeError, ValueError):
        pass
    logprobs = np.concatenate([np.asarray(out["logprobs"]) for out in result.loss_fn_outputs])
    weights = np.concatenate([np.asarray(d.loss_fn_inputs["weights"], dtype=np.float64) for d in datums])
    total = weights.sum()
    return float(-(logprobs * weights).sum() / total) if total else float("nan")


def inspect(path: str, limit: int | None) -> None:
    rows = _load_rows(path, limit)
    lengths = [len(row["user"]) + len(row["assistant"]) for row in rows]
    print(json.dumps({
        "rows": len(rows),
        "runs": sorted({row["run"] for row in rows}),
        "sources": {name: sum(row["source"] == name for row in rows) for name in ("rule", "llm")},
        "chars_min": min(lengths),
        "chars_max": max(lengths),
        "chars_mean": round(sum(lengths) / len(lengths), 1),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="training JSONL")
    parser.add_argument("--val-data", help="optional validation JSONL (loss only)")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="only use the first N training rows")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--save-name", default="planner-sft-v1")
    parser.add_argument("--save-state-every", type=int, default=0, help="also save a Train checkpoint every N steps")
    parser.add_argument("--resume", help="save_state() path; restores weights and optimizer")
    parser.add_argument("--resume-weights-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="validate data without contacting PyTRIO")
    parser.add_argument("--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--swanlab-project", default="zhixun-planner-sft")
    parser.add_argument("--swanlab-run", default=None)
    args = parser.parse_args()

    if args.inspect:
        if not args.data:
            parser.error("--inspect needs --data")
        inspect(args.data, args.limit)
        return

    import pytrio as trio

    service = trio.ServiceClient()
    if args.list_models:
        for index, name in enumerate(service.get_supported_models(), start=1):
            print(f"{index}. {name}")
        return
    if not args.data:
        parser.error("--data is required")

    rows = _load_rows(args.data, args.limit)
    val_rows = _load_rows(args.val_data) if args.val_data else []
    print(json.dumps({"train_rows": len(rows), "val_rows": len(val_rows), "base_model": args.base_model,
                      "rank": args.rank, "epochs": args.epochs, "batch_size": args.batch_size,
                      "lr": args.lr, "max_seq_len": args.max_seq_len}, ensure_ascii=False))

    training = service.create_lora_training_client(base_model=args.base_model, rank=args.rank, seed=args.seed)
    tokenizer = training.get_tokenizer()
    print("tokenizer loaded")

    def make(rows_):
        datums, skipped = [], {}
        for row in rows_:
            datum, reason = render(row, tokenizer, args.max_seq_len)
            if datum is None:
                skipped[reason] = skipped.get(reason, 0) + 1
            else:
                datums.append(datum)
        return datums, skipped

    train_datums, skipped = make(rows)
    val_datums, val_skipped = make(val_rows) if val_rows else ([], {})
    print(json.dumps({"train_datums": len(train_datums), "train_skipped": skipped,
                      "val_datums": len(val_datums), "val_skipped": val_skipped}, ensure_ascii=False))
    if not train_datums:
        raise SystemExit("no trainable rows after length filtering")

    if args.resume:
        (training.load_state if args.resume_weights_only else training.load_state_with_optimizer)(args.resume).result()
        print(f"resumed from {args.resume}")

    swan = None
    if args.swanlab_mode != "disabled":
        import swanlab
        swan = swanlab.init(
            project=args.swanlab_project,
            name=args.swanlab_run or args.save_name,
            mode=args.swanlab_mode,
            config={"base_model": args.base_model, "rank": args.rank, "epochs": args.epochs,
                    "batch_size": args.batch_size, "lr": args.lr, "rows": len(train_datums),
                    "data": args.data, "max_seq_len": args.max_seq_len},
        )

    rng = random.Random(args.seed)
    steps_per_epoch = max(1, math.ceil(len(train_datums) / args.batch_size))
    started = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train_datums)))
        rng.shuffle(order)
        epoch_loss, epoch_tokens = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [train_datums[i] for i in order[start:start + args.batch_size]]
            fwdbwd = training.forward_backward(batch, "cross_entropy")
            optim = training.optim_step(trio.AdamParams(learning_rate=args.lr))
            result = fwdbwd.result()
            optim.result()
            loss = _per_token_loss(result, batch)
            tokens = int(sum(len(d.loss_fn_inputs["weights"]) for d in batch))
            global_step += 1
            epoch_loss += loss * tokens
            epoch_tokens += tokens
            record = {"step": global_step, "epoch": epoch + 1, "loss": round(loss, 5),
                      "tokens": tokens, "elapsed_s": round(time.time() - started, 1)}
            print(json.dumps(record, ensure_ascii=False))
            if swan is not None:
                swan.log({"train/loss": loss, "train/tokens": tokens, "train/epoch": epoch + 1},
                         step=global_step)
            if args.save_state_every and global_step % args.save_state_every == 0:
                saved_state = training.save_state(name=f"{args.save_name}-step{global_step}").result()
                print(json.dumps({"train_state": saved_state.path}, ensure_ascii=False))
        mean_loss = epoch_loss / epoch_tokens if epoch_tokens else float("nan")
        if swan is not None:
            swan.log({"train/epoch_loss": mean_loss}, step=global_step)
        print(json.dumps({"epoch": epoch + 1, "epoch_loss": round(mean_loss, 5),
                          "steps": steps_per_epoch, "elapsed_s": round(time.time() - started, 1)},
                         ensure_ascii=False))
        if val_datums:
            val_losses, val_tokens = 0.0, 0
            for start in range(0, len(val_datums), args.batch_size):
                batch = val_datums[start:start + args.batch_size]
                result = training.forward_backward(batch, "cross_entropy").result()
                loss = _per_token_loss(result, batch)
                tokens = int(sum(len(d.loss_fn_inputs["weights"]) for d in batch))
                val_losses += loss * tokens
                val_tokens += tokens
            val_loss = val_losses / val_tokens if val_tokens else float("nan")
            print(json.dumps({"epoch": epoch + 1, "val_loss": round(val_loss, 5)}, ensure_ascii=False))
            if swan is not None:
                swan.log({"val/loss": val_loss}, step=global_step)

    saved = training.save_weights_for_sampler(name=args.save_name).result()
    print(json.dumps({"saved_path": saved.path, "saved_model": getattr(saved, "model", None),
                      "save_name": args.save_name, "elapsed_s": round(time.time() - started, 1)},
                     ensure_ascii=False, indent=2))
    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
