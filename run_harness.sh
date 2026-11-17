#!/usr/bin/env bash
# Start the closed loop (drift -> diagnose -> experiment -> gate -> rollback).
#
#   ./run_harness.sh
#   RUL_DATA_ROOT=/path/to/pack ./run_harness.sh
#   LLM_MODE=policy MAX_EXPERIMENTS=3 ./run_harness.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
WORKSPACE="${WORKSPACE:-runs/harness-b}"
BATCH_SIZE="${BATCH_SIZE:-200}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-6}"
LLM_MODE="${LLM_MODE:-auto}"

if [[ -n "${RUL_DATA_ROOT:-}" ]]; then
  echo "data root: $RUL_DATA_ROOT"
fi

rm -rf "$WORKSPACE"
"$PYTHON" -m harness.run \
  --workspace "$WORKSPACE" \
  --batch-size "$BATCH_SIZE" \
  --max-experiments "$MAX_EXPERIMENTS" \
  --llm-mode "$LLM_MODE"

echo
echo "artefacts: $WORKSPACE/{run_report.json,ledger.jsonl,swanlab.jsonl}"
echo "next:      $PYTHON tools/show_run.py --workspace $WORKSPACE"
