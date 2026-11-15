# Skill: safety, gating and rollback

The system is judged on controlled self-improvement, not on aggressive updates.

- The champion is only replaced when the gate reports `promote`. The gate
  requires a mean improvement **and** no single-bearing regression above
  tolerance; a mean win that hides a local failure is rejected.
- Keep the last champion checkpoint and its digest. `rollback` restores it
  byte-for-byte and is always available.
- Budget limits are hard: when `experiments_left` or `train_seconds_left`
  reaches zero the harness degrades to `do_nothing`, even if a model proposes
  an experiment.
- Any planner output that fails schema validation is discarded and replaced by
  the deterministic policy; the substitution is written to the decision record
  so a reviewer can see when and why the fallback fired.
- Nothing in the loop opens a network connection except the optional prediction
  call to the local model endpoint; training and scoring are fully offline.
