# Skill: experiment design under a budget

Every experiment must be falsifiable and cheap enough that a failure is cheap.

- One variable per cycle. Do not change width, schedule and features at once.
- State the hypothesis as a prediction: "refitting the head on the public replay
  buffer will recover at least 1% RMSE because the batch shows a constant
  offset". If you cannot write the prediction, choose `do_nothing`.
- Order by cost: `do_nothing` < `rollback` < `recalibrate_inputs` <
  `retrain_head` < `hpo`. Do not skip a level without evidence that it failed.
- Keep at least one experiment in reserve until the last batch.
- Record the cost with the result: training seconds, number of trials, tokens.
  A gain that costs the whole budget is a regression in the scored objective
  (unit gain per unit resource).
- If a candidate is rejected, the next cycle must not repeat the same action
  with the same parameters. Change the hypothesis or stop.
