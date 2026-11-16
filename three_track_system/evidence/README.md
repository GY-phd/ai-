# Evidence · one real closed-loop run

Everything here is produced by `python -m harness.run`; nothing was hand-written.

| File | What it is |
|---|---|
| `ledger.jsonl` | Decision ledger, **hash-chained**: each record stores `prev_sha256` (previous record's digest, `"genesis"` for the first) and its own `record_sha256` |
| `swanlab.jsonl` | Metric stream in SwanLab shape (per batch, per cycle) |
| `run_report.json` | Run summary: baseline/final metrics, budget consumption, decisions, champion history |
| `hypothesis_chain.jsonl` | The four-stage evidence chain: read history → hypothesis → experiment → decision, one record per cycle (including the cycles that decided to change nothing) |
| `registry/champion.json` | Current champion pointer + metrics |
| `registry/champions.jsonl` | Champion version chain (baseline → 301.18 → 283.54) |
| `champion_checkpoint.npz` | Copy of the promoted checkpoint so the registry pointer resolves inside the archive |

Verify the chain:

```bash
python tools/verify_ledger.py --ledger evidence/ledger.jsonl
# OK  9 records, 9 decisions, 2 promotions, chain intact
```

## The run this evidence comes from

| | |
|---|---|
| Baseline (official static model) | RMSE 681.160 / MAE 563.106 |
| Final champion | **RMSE 283.539 / MAE 218.248** |
| Improvement | **−58.37 %** |
| Cycles | 9 (2 promote, 1 reject, 6 do-nothing) |
| Parameters | 3,457 (well under the 100,000 ceiling) |
| Planner | deterministic policy (no LLM endpoint reachable) |
| Checkpoint | `contest.cpu-mlp.v1` — **verified loadable by the official `baselines/model.py::Model.load`** |

## About the absolute paths inside `ledger.jsonl`

The ledger records the checkpoint path exactly as it existed on the machine that
ran the experiment (`D:\...\runs\harness-fused\experiments\...`). Those strings are
left untouched on purpose: rewriting any byte of a record would invalidate the
hash chain, and the whole point of the chain is that the record cannot be edited
after the fact. The paths reveal nothing beyond a local directory name.

The copied `registry/champion.json` is **not** hash-locked, so its `checkpoint`
field *was* rebased to `../champion_checkpoint.npz` to keep the archive
self-contained; its metrics and digest are unchanged.
