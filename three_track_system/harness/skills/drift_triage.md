# Skill: drift triage

Use this when `DriftMonitor` returns `warn` or `alarm` for the current batch.

1. Confirm the signal is about inputs, not about the scoring pipeline. A batch
   with degenerate channels (constant std) is a data fault: choose
   `do_nothing` and flag it.
2. Compare `psi_max` with `share_shifted`:
   - high PSI on a few features, low share -> localised shift, usually an
     operating-condition change. Input calibration is cheap and reversible.
   - high share across most features -> the whole operating point moved; the
     mapping is likely intact, so recalibrating inputs is still the first move.
3. Only claim `concept` drift when the error grew *faster* than the input shift
   and the residual is not explained by a constant offset. A pure offset with
   preserved ordering is a prior problem, not a concept problem: refit the head
   before paying for retraining.
4. Never retrain on a batch whose labels have not arrived.

Evidence to cite in the decision record: PSI max, share shifted, batch RMSE
versus champion RMSE, and the top-3 shifted feature names.
