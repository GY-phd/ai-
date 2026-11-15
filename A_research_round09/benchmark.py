"""Cost only: fixed fold-0 models, with no parameter or model selection."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter, process_time

from study import ROOT, TRAIN_SHA, StructuredModel, anchor_update, ex, np, shift
from A_research_round08.study import RawSupport, SEED, cost, process_peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = ex.read_dataset(args.train, "classification", ex.CLASSES)
    assert train["sha256"] == TRAIN_SHA
    record = json.loads((args.run / "results.json").read_text())["updates"][0]
    fit = np.zeros(len(train["y"]), dtype=bool)
    fit[record["partition"]["fit"]] = True
    evaluation = record["partition"]["evaluation"]
    x, y = train["x"], train["y"]
    measured = {}

    def timed_fit(name, function):
        cpu, wall = process_time(), perf_counter()
        model = function()
        measured[name] = {"fit_cpu_seconds": process_time() - cpu,
                          "fit_wall_seconds": perf_counter() - wall}
        return model

    stable = timed_fit("stable", lambda: ex.source_control(train, fit, SEED))
    raw = timed_fit("raw_support", lambda: RawSupport(stable, x[fit], y[fit]))
    parent = timed_fit("frozen_A", lambda: StructuredModel.fit(x[fit], y[fit]))
    rows = record["budgets"]["2"]["calibration_rows"]
    cx = shift(x[rows], "composite")
    model = timed_fit("C_replay_anchor_12", lambda: anchor_update(parent, cx, y[rows], x[fit], y[fit]))
    expected = StructuredModel.load(args.run / "f0_budget2.npz")
    np.testing.assert_allclose(model.coef, expected.coef, rtol=1e-9, atol=1e-10)
    cpu, wall = process_time(), perf_counter()
    for _ in range(100):
        anchor_update(parent, cx, y[rows], x[fit], y[fit])
    measured["C_replay_anchor_12"]["update_100_repeat_mean"] = {
        "cpu_ms": (process_time() - cpu) * 10, "wall_ms": (perf_counter() - wall) * 10}
    for name, candidate in (("stable", stable), ("raw_support", raw), ("frozen_A", parent), ("C_replay_anchor_12", model)):
        measured[name]["inference"] = cost(candidate, x[evaluation])
    result = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "platform": platform.platform(), "processor": platform.processor()},
              "models": measured, "replay_rows": int(sum(fit)),
              "replay_input_bytes": int(x[fit].nbytes + y[fit].nbytes),
              "process_peak_working_set_bytes": process_peak(), "llm_calls": 0, "llm_tokens": 0,
              "notes": ["preloaded model latency excludes files/SQLite", "raw_support fit excludes its shared stable parent",
                        "single-call CPU may round to zero; repeated measurements reduce timer quantization",
                        "repeat 100 times for timing only, always from the same parent"]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
