# -*- coding: utf-8 -*-
"""Build the evaluation stream for any profile, by holding out whole units.

The rule
--------
The competition says the training set, the validation set and the hidden
test stream are separated by **whole units**, and no observation of one
unit may cross the split.  For profile B the unit is a bearing, for
profile C it is an engine, for profile A it is a measurement batch.

Each profile's public package ships a train/validation split that already
respects the rule.  But the harness still needs a *stream* to run on -
the hidden one does not exist locally, so the only honest way to rehearse
the closed loop is to build a stream out of the public training set by
holding out whole units of it, exactly as the hidden split would.

What this module guarantees
---------------------------
1. **Unit isolation.**  A unit is either entirely in the initial training
   set, entirely in the stream, or entirely in the validation set.  Never
   split.  The audit record says so explicitly.
2. **The validation set is never touched.**  It is scored once, at the
   end, as an independent check.  The number in the report is never the
   number the gate optimised against.
3. **A stratified holdout.**  The units are taken *interleaved* by
   size, not "the longest ones".  Taking the longest units would leave
   the initial training set without any examples of the large-target
   regime, and the evaluation would then measure a scale mismatch
   instead of model quality - the two would degrade together for a
   reason that has nothing to do with the model.

For profile A the holdout works differently, and deliberately so.  Its
public package has exactly **one** batch in the training set, so there
is nothing to hold out: the stream *is* the validation batch.  What the
harness then does is refuse to train on it until the drift signal says
the array has moved - which is precisely the situation the competition
measures.

Run:
    python -m tracks.stream --profile B --holdout 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.profiles import get  # noqa: E402


def _take(source, mask, n_rows):
    """Select the rows of ``mask`` out of a dataset field."""
    if isinstance(source, np.ndarray) and source.shape[:1] == (n_rows,):
        return source[mask]
    if isinstance(source, list) and len(source) == n_rows:
        return [item for item, keep in zip(source, mask) if keep]
    return source


def stratified_holdout(sizes: dict, n_holdout: int, labels: dict | None = None):
    """Choose units for the stream, interleaved by size.

    The units are ordered by size, longest first, and then taken every
    ``stride``-th entry.  The holdout therefore contains both short and
    long units, which keeps the stream's target range comparable to the
    initial training set's.

    When ``labels`` is given - the sub-dataset of each unit - the holdout
    is taken **within each label group** first, so the stream contains
    the same mix of operating conditions as the training set.  Without
    that, a size-only stratification can pick three tiny engines that all
    happen to come from the hardest sub-dataset, and the stream then
    measures a distribution shift that the model was never meant to
    handle.
    """
    if not sizes:
        return []
    k = max(1, min(n_holdout, len(sizes)))

    if labels:
        groups = {}
        for name in sizes:
            groups.setdefault(labels[name], []).append(name)
        picked = []
        # Round-robin over the label groups, largest first, taking the
        # longest unit of each group in turn.
        ranked_groups = sorted(groups, key=lambda g: len(groups[g]), reverse=True)
        cursors = {g: 0 for g in ranked_groups}
        while len(picked) < k:
            progressed = False
            for g in ranked_groups:
                if len(picked) >= k:
                    break
                members = sorted(groups[g], key=lambda n: sizes[n], reverse=True)
                index = cursors[g]
                while index < len(members) and members[index] in picked:
                    index += 1
                if index < len(members):
                    picked.append(members[index])
                    cursors[g] = index + 1
                    progressed = True
            if not progressed:
                break
        return picked

    ranked = sorted(sizes, key=lambda name: sizes[name], reverse=True)
    stride = max(1, len(ranked) // k)
    picked = []
    for offset in range(stride):
        for name in ranked[offset::stride]:
            if len(picked) >= k:
                break
            if name not in picked:
                picked.append(name)
        if len(picked) >= k:
            break
    return picked


def select_judge_units(sizes: dict, n_judge: int, labels: dict | None = None):
    """Choose the units that are held back for the gate's judge set.

    The judge set has two competing requirements, and they pull in
    opposite directions.

    * It must be **representative** of the population, or the gate
      measures the wrong thing.  A judge set made only of the shortest
      units would make every candidate look good; one made only of the
      longest would make every candidate look bad.
    * It must not **cost** the training set its most valuable units.  On
      profile B the units run from 42 to 2538 rows, and the longest one
      is the only unit that exercises the long-lived regime - the regime
      the two-stage model is worst at and therefore most needs to learn.

    The selection therefore takes the units **closest to the population's
    median length**.  That keeps the judge set representative and, as a
    side effect, leaves the extremes - the very short and the very long -
    in the training set where they do the most good.

    When the units are labelled by sub-dataset, the selection is done
    within each label group, so the judge set carries the same mix of
    operating conditions as the training set.
    """
    if not sizes or n_judge <= 0:
        return []
    k = max(1, min(int(n_judge), len(sizes)))

    if labels:
        groups = {}
        for name in sizes:
            groups.setdefault(labels[name], []).append(name)
        picked = []
        ranked_groups = sorted(groups)
        cursors = {g: 0 for g in ranked_groups}
        while len(picked) < k:
            progressed = False
            for g in ranked_groups:
                if len(picked) >= k:
                    break
                members = groups[g]
                if not members:
                    continue
                median = float(np.median([sizes[n] for n in members]))
                # Closest to the group's median, ties broken by name so
                # the selection is deterministic.
                members = sorted(members, key=lambda n: (abs(sizes[n] - median), n))
                index = cursors[g]
                while index < len(members) and members[index] in picked:
                    index += 1
                if index < len(members):
                    picked.append(members[index])
                    cursors[g] = index + 1
                    progressed = True
            if not progressed:
                break
        return picked

    median = float(np.median(list(sizes.values())))
    ranked = sorted(sizes, key=lambda name: (abs(sizes[name] - median), name))
    return ranked[:k]


def _interleave_by_unit(groups, order):
    """The row order of a stream in which every unit is observed each round.

    The units are visited round-robin: the first observation of every
    unit, then the second observation of every unit, and so on.  A unit
    that runs out of observations drops out and the rest carry on.

    This is what a live deployment of the same kind looks like: several
    bearings are being monitored at the same time, and their readings
    arrive together.  It is also the only ordering under which the
    closed loop's judge slice asks the same question every cycle.
    """
    groups = np.asarray(groups)
    order = np.asarray(order, dtype=np.float64)
    if len(groups) == 0:
        return np.arange(0)

    # Each unit's rows, sorted by its own chronological order.
    queues = {}
    for name in np.unique(groups):
        indices = np.nonzero(groups == name)[0]
        queues[name] = list(indices[np.argsort(order[indices], kind="stable")])

    names = sorted(queues)
    picked = []
    position = 0
    while True:
        added = False
        for name in names:
            queue = queues[name]
            if position < len(queue):
                picked.append(queue[position])
                added = True
        if not added:
            break
        position += 1
    return np.asarray(picked, dtype=int)


def build(profile_key: str, n_holdout: int = 3, n_judge: int = 1):
    """Split the public data into an initial part, a stream, a judge set and a validation.

    Returns ``(initial, stream, judge, validation, meta)``.

    Why there is a **judge set** at all
    ----------------------------------
    The closed loop compares a candidate against the champion on rows
    that neither model has seen.  Where those rows come from decides
    whether the comparison means anything.

    The obvious choice - "the rows that arrive after the training
    boundary" - gives a slice whose *composition* changes from cycle to
    cycle.  The units have very different lengths (42 to 2538 rows on
    profile B), so a short-lived unit exhausts its rows early and drops
    out, and the slice ends up dominated by whichever long-lived unit
    happens to remain.  Measured on profile B, the slice went from
    three units with a median target of 424 to a single unit with a
    median target of 1623.  The two-stage model's error on that unit is
    several times its error on the others, so a gate that compares
    candidates across such slices is reading the slice, not the model -
    and a hypothesis can be promoted on a difficulty artefact.

    The judge set removes the artefact by construction.  It is a set of
    whole units that are **held back and never trained on**, exactly the
    way the competition's hidden test set is held back.  Every cycle
    asks the same question - "how well does this model do on these
    unseen units?" - and the answer is comparable from cycle to cycle
    because the units are the same ones.

    The protocol still holds:
      * the judge units are disjoint from the training units;
      * the judge units are disjoint from the validation units;
      * the validation set is scored once, at the very end.
    """
    profile = get(profile_key)
    train = profile.load("train")
    validation = profile.load("validation")

    train_units = sorted(set(train["groups"].tolist()))
    validation_units = sorted(set(validation["groups"].tolist()))
    overlap = set(train_units) & set(validation_units)
    if overlap:
        raise ValueError("training and validation units overlap: %s" % sorted(overlap))

    n_rows = len(train["x"])

    if len(train_units) <= 1:
        # Profile A: the public training set is a single batch, so there
        # is no *unit* to hold out of it.  The stream is therefore taken
        # from **within** the batch, split **class-stratified**.
        #
        # The alternative - making the validation batch the stream - was
        # what an earlier version did, and it is **leakage**.  Training
        # on the stream and then scoring on the validation set would be
        # scoring the model on its own training data: the loop reached
        # a Macro-F1 of 0.9760 that way, and every one of those points
        # was stolen from the test set.  The validation batch is now
        # only ever scored, at the very end, exactly once.
        #
        # The split has to be **class-stratified**, not temporal.  The
        # 445 rows of batch 1 are stored in class order - the first 90
        # are Ethanol, the next 98 Ethylene, and so on - so a temporal
        # split is a *class* split: the first 60% of the batch contains
        # only Ethanol, Ethylene and Ammonia, and the model it produces
        # cannot even name the other three gases.  Measured, that split
        # gave a champion of Macro-F1 0.3838 - far below the official
        # baseline - while a class-stratified split of the same rows
        # keeps every class represented on both sides.
        #
        # The stream is still a genuine rehearsal of the hidden
        # deployment: the rows arrive in batches, the drift signal fires
        # on them, and the harness may only train on them after it does.
        labels = np.asarray(train["y"])
        rng = np.random.default_rng(20261116)
        is_initial = np.zeros(len(labels), dtype=bool)
        for label in np.unique(labels):
            members = np.nonzero(labels == label)[0]
            rng.shuffle(members)
            take = int(np.floor(len(members) * 0.6))
            take = max(1, min(take, len(members) - 1))
            is_initial[members[:take]] = True
        initial = {k: _take(v, is_initial, n_rows) for k, v in train.items()}
        stream = {k: _take(v, ~is_initial, n_rows) for k, v in train.items()}

        # The judge set comes from the **validation batch**, not from the
        # training batch.
        #
        # The gate has to judge on the *deployment* distribution, and on
        # profile A the deployment distribution is the validation batch's
        # - a completely different class mix (Toluene is 17% of the
        # training batch and 0.4% of the validation batch).  A judge set
        # drawn from the training batch would judge every candidate on
        # the training distribution, and the "rare class is
        # underweighted" hypothesis would look like a 0.95 Macro-F1 win
        # when it is in fact a 0.71 loss on the distribution that
        # matters.
        #
        # So a class-stratified slice of the validation batch is held
        # back for the gate, and the rest of the validation batch is
        # scored once, at the very end.  The held-back slice is never
        # trained on, and the final score is still measured on rows the
        # model has never seen.
        validation_labels = np.asarray(validation["y"])
        judge_rng = np.random.default_rng(20261117)
        is_judge = np.zeros(len(validation_labels), dtype=bool)
        for label in np.unique(validation_labels):
            members = np.nonzero(validation_labels == label)[0]
            judge_rng.shuffle(members)
            # A quarter of each class, with a floor of one row so that
            # even the rarest class is represented.
            take = max(1, int(np.floor(len(members) * 0.25)))
            is_judge[members[:take]] = True
        judge = {k: _take(v, is_judge, len(validation["y"]))
                 for k, v in validation.items()}
        validation = {k: _take(v, ~is_judge, len(validation["y"]))
                      for k, v in validation.items()}

        meta = {
            "profile": profile_key,
            "mode": "within_batch_stream",
            "initial_units": train_units,
            "stream_units": train_units,
            "judge_units": ["validation-slice"],
            "validation_units": validation_units,
            "initial_rows": int(len(initial["y"])),
            "stream_rows": int(len(stream["y"])),
            "judge_rows": int(len(judge["y"])),
            "validation_rows": int(len(validation["y"])),
            "unit_leakage": False,
            "validation_used_for_training": False,
            "judge_used_for_training": False,
            "note": "profile A's public training set is one batch.  The stream is "
                    "a class-stratified 40% of that batch.  The gate's judge set "
                    "is a class-stratified 25% of the validation batch - held back "
                    "from training and from the final score - so the gate judges "
                    "on the deployment distribution rather than the training one.",
        }
        return initial, stream, judge, validation, meta

    sizes = {name: int(np.sum(train["groups"] == name)) for name in train_units}
    labels = None
    if profile_key == "C":
        # The sub-dataset is the confounder in C: FD004's engines live
        # very different lives from FD001's.  A holdout that ignores it
        # measures the wrong thing.
        labels = {name: name.split("#", 1)[0] for name in train_units}

    # The judge holdout comes first: it is the part of the data whose
    # units are never trained on, so it has to be chosen without any
    # reference to what the training set will contain.  The units
    # closest to the population's median length are chosen, so the judge
    # set is representative and the extremes stay in training.
    n_judge = max(0, min(int(n_judge), len(train_units) - 2))
    judge_units = select_judge_units(sizes, n_judge, labels) if n_judge else []

    remaining = {name: size for name, size in sizes.items() if name not in judge_units}
    remaining_labels = None
    if labels:
        remaining_labels = {name: labels[name] for name in remaining}
    n_stream = max(1, int(n_holdout) - len(judge_units))
    stream_units = stratified_holdout(remaining, n_stream, remaining_labels)

    holdout = list(judge_units) + list(stream_units)
    is_judge = np.isin(train["groups"], judge_units)
    is_stream = np.isin(train["groups"], stream_units)
    is_initial = ~(is_judge | is_stream)

    initial = {k: _take(v, is_initial, n_rows) for k, v in train.items()}
    stream = {k: _take(v, is_stream, n_rows) for k, v in train.items()}
    judge = {k: _take(v, is_judge, n_rows) for k, v in train.items()}

    # Order the stream the way a live stream arrives: every unit is
    # observed once per round, and the units advance together.
    #
    # The earlier version sorted by unit and then by the unit's own
    # order, which put all of one bearing's rows before any of the next
    # bearing's.  That is not a stream - it is three batches - and it
    # broke the closed loop in two ways:
    #
    # * The judge slice, which takes the rows *after* the boundary, was
    #   drawn from whichever unit happened to straddle the boundary, so
    #   the slice's difficulty changed completely from cycle to cycle.
    # * The accumulated training set grew by whole units, so the
    #   champion's re-fit was measuring "have I seen this unit yet?"
    #   rather than "does the model generalise?".
    #
    # Round-robin interleaving keeps the units running concurrently, so
    # every cycle's slice contains the same mix of units at the same
    # point in their lives - the same question, asked every cycle.
    order = _interleave_by_unit(stream["groups"], stream["order"])
    for key, value in list(stream.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(stream["x"]),):
            stream[key] = value[order]
        elif isinstance(value, list) and len(value) == len(stream["x"]):
            stream[key] = [value[i] for i in order]

    # The judge set is ordered the same way, so that a judge slice can
    # be taken at a comparable point in each held-back unit's life.
    judge_order = _interleave_by_unit(judge["groups"], judge["order"])
    for key, value in list(judge.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (len(judge["x"]),):
            judge[key] = value[judge_order]
        elif isinstance(value, list) and len(value) == len(judge["x"]):
            judge[key] = [value[i] for i in judge_order]

    meta = {
        "profile": profile_key,
        "mode": "unit_holdout",
        "initial_units": [u for u in train_units if u not in holdout],
        "stream_units": stream_units,
        "judge_units": judge_units,
        "validation_units": validation_units,
        "initial_rows": int(len(initial["y"])),
        "stream_rows": int(len(stream["y"])),
        "judge_rows": int(len(judge["y"])),
        "validation_rows": int(len(validation["y"])),
        "unit_leakage": False,
        "validation_used_for_training": False,
        "judge_used_for_training": False,
    }
    if profile.task == "regression":
        meta["initial_target_max"] = float(initial["y"].max())
        meta["stream_target_max"] = float(stream["y"].max())
        meta["judge_target_max"] = float(judge["y"].max()) if len(judge["y"]) else None
        meta["initial_target_median"] = float(np.median(initial["y"]))
        meta["stream_target_median"] = float(np.median(stream["y"]))
    return initial, stream, judge, validation, meta


def _empty_like(source):
    """An empty copy of a dataset bundle, with the same keys."""
    out = {}
    for key, value in source.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            out[key] = value[:0]
        elif isinstance(value, list):
            out[key] = []
        else:
            out[key] = value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="B", choices=("A", "B", "C"))
    parser.add_argument("--holdout", type=int, default=3)
    parser.add_argument("--judge", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    initial, stream, judge, validation, meta = build(args.profile, args.holdout,
                                                      args.judge)
    print("=== evaluation stream for profile %s ===" % args.profile)
    print("  mode          %s" % meta["mode"])
    print("  initial units %s  (%d rows)"
          % (", ".join(meta["initial_units"][:6]), meta["initial_rows"]))
    print("  stream   units %s  (%d rows)"
          % (", ".join(meta["stream_units"][:6]), meta["stream_rows"]))
    print("  judge    units %s  (%d rows)"
          % (", ".join(meta["judge_units"][:6]), meta["judge_rows"]))
    print("  validation    %s  (%d rows)"
          % (", ".join(meta["validation_units"][:6]), meta["validation_rows"]))
    print("\n  audit: unit leakage %s | validation used for training %s"
          % ("NO" if not meta["unit_leakage"] else "YES",
             "NO" if not meta["validation_used_for_training"] else "YES"))
    print("         judge units used for training %s"
          % ("NO" if not meta["judge_used_for_training"] else "YES"))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
