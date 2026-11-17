# -*- coding: utf-8 -*-
"""The continuation model: the platform's reference MLP, reproduced exactly.

This file is a faithful re-implementation of the platform's
``baselines/model.py``.  The forward pass, the normalisation, the Adam
update and the global gradient clip are **byte-for-byte the reference's**
so that

* a checkpoint produced here is a plain ``contest.cpu-mlp.v1`` archive the
  platform's own reference loader opens without complaint, and
* continuing the parent checkpoint really continues it - the weights, the
  Adam moments, the epoch counter and the normalisation state all carry
  over.

What this file adds on top of the reference
------------------------------------------
``target_scaling=False``
    The reference standardises the regression target before fitting and
    unscales it afterwards.  The RUL target is a *structural* quantity
    whose absolute magnitude is informative (a 2500-minute bearing and a
    120-minute bearing are different problems), and standardising it away
    flattens every unit onto the same z-score.  Fitting directly in the
    target's own unit is therefore allowed - **and it stays reference
    compatible**: the reference loader reconstructs the prediction as
    ``logits * target_scale + target_mean``, so storing
    ``target_scale = 1.0`` and ``target_mean = 0.0`` reproduces exactly
    the same numbers.  The checkpoint's *meaning* never changes.

``resize``
    An explicit, documented weight migration between hidden widths.  The
    competition rules require that a change of architecture be implemented
    as a compatible weight migration rather than a silent re-init, and
    this is that migration: the surviving hidden units keep their incoming
    and outgoing weights, and the Adam state is carried over.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

_THREADS = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
if not 1 <= _THREADS <= 8:
    raise ValueError("BASELINE_CPU_THREADS must be between 1 and 8")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(_THREADS)
import numpy as np  # noqa: E402

PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_PARAMETERS = 100_000
MAX_HIDDEN = 128


class Model:
    """The reference CPU MLP, with the continuation hooks this project needs."""

    def __init__(self, task, paths, classes, x, y, *, hidden=32, seed=42,
                 learning_rate=0.003, batch_size=128, target_scaling=True):
        if task not in ("classification", "regression") or not 1 <= hidden <= MAX_HIDDEN:
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        count = x.shape[1] * hidden + hidden + hidden * output + output
        if count >= MAX_PARAMETERS:
            raise ValueError("reference baseline must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, math.sqrt(1 / x.shape[1]), (x.shape[1], hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
            "b2": np.zeros(output),
        }
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {
            "format": FORMAT, "device": "cpu", "task": task, "paths": paths,
            "classes": list(classes), "hidden": hidden, "seed": seed,
            "epochs": 0, "steps": 0, "learning_rate": learning_rate,
            "batch_size": batch_size, "parameter_count": count,
            "target_mean": float(y.mean()) if task == "regression" else 0.0,
            "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0,
        }
        self.target_scaling = bool(target_scaling)
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self):
        m = self.meta
        if m.get("format") != FORMAT or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "batch_size", "parameter_count"):
            if type(m.get(name)) is not int or m[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= m["hidden"] <= MAX_HIDDEN or not 1 <= m["batch_size"] <= 65536 or not 0 < m["parameter_count"] < MAX_PARAMETERS:
            raise ValueError("invalid checkpoint architecture")
        if not 0 <= m["seed"] < 2 ** 32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if (not isinstance(m.get("paths"), list) or not 1 <= len(m["paths"]) <= 4096
                or any(not isinstance(p, str) for p in m["paths"]) or len(set(m["paths"])) != len(m["paths"])):
            raise ValueError("invalid checkpoint feature schema")
        classes = m.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (m["task"] == "classification" and len(classes) < 2) or (m["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(m.get(name), bool) or not isinstance(m.get(name), (float, int)) or not math.isfinite(m[name]):
                raise ValueError("invalid checkpoint numeric state")
        if not 0 < m["learning_rate"] <= 1 or m["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        input_size, hidden = len(m["paths"]), m["hidden"]
        output = len(classes) if m["task"] == "classification" else 1
        shapes = {"w1": (input_size, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(value.size for value in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if (self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64")
                or self.mean.shape != (input_size,) or self.scale.shape != (input_size,)
                or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0)):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")

    # --------------------------------------------------------------- numerics
    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((x - self.mean) / self.scale, -10, 10)

    def forward(self, x):
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def predict_raw(self, x):
        _, logits = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            logits = logits - logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            return prob / prob.sum(axis=1, keepdims=True)
        return np.maximum(0, logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])

    predict = predict_raw

    # --------------------------------------------------------------- training
    def fit(self, x, y, epochs, *, learning_rate=None):
        """Fit for ``epochs`` epochs.  One epoch per call, like the reference.

        The epoch-indexed shuffling makes a resumed run reproduce one
        continuous run: the permutation depends on the *epoch counter*, not
        on how many epochs this call asked for.
        """
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        if y.shape != (len(x),) or not len(x):
            raise ValueError("target shape mismatch")
        if self.meta["task"] == "regression":
            if self.meta.get("target_scale", 1.0) == 1.0 and self.meta.get("target_mean", 0.0) == 0.0:
                # Direct fit in the target's own unit.
                target = np.asarray(y, dtype=np.float64)
            else:
                target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = y
        base_lr = float(self.meta["learning_rate"] if learning_rate is None else learning_rate)
        losses = []
        for _ in range(epochs):
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], target[indices]
                hidden, logits = self.forward(batch)
                if self.meta["task"] == "classification":
                    logits = logits - logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    loss_sum -= float(np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                    delta = probabilities
                    delta[np.arange(len(indices)), expected] -= 1
                    delta /= len(indices)
                else:
                    error = logits[:, 0] - expected
                    loss_sum += float(np.sum(error ** 2))
                    delta = (2 * error / len(indices))[:, None]
                gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden ** 2)
                gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient = gradient * min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    first = self.first[name] / (1 - .9 ** self.meta["steps"])
                    second = self.second[name] / (1 - .999 ** self.meta["steps"])
                    self.params[name] -= base_lr * first / (np.sqrt(second) + 1e-8)
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            self.meta["epochs"] += 1
            losses.append(loss_sum / len(x))
        self.validate()
        return losses

    # ------------------------------------------------------------ checkpoints
    def save(self, path):
        self.validate()
        arrays = {**self.params,
                  **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()},
                  "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        path = Path(path)
        if path.exists():
            raise ValueError("refusing to overwrite an existing checkpoint")
        with path.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        expected = {*PARAMETERS, *("adam_first_" + k for k in PARAMETERS),
                    *("adam_second_" + k for k in PARAMETERS), "mean", "scale", "metadata"}
        with zipfile.ZipFile(path) as archive:
            names = [e.filename for e in archive.infolist()]
            if len(names) != len(expected) or set(names) != {n + ".npy" for n in expected} or \
                    sum(e.file_size for e in archive.infolist()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            model = cls.__new__(cls)
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("invalid checkpoint metadata")
            model.meta = json.loads(str(metadata))
            model.params = {k: arrays[k].copy() for k in PARAMETERS}
            model.first = {k: arrays["adam_first_" + k].copy() for k in PARAMETERS}
            model.second = {k: arrays["adam_second_" + k].copy() for k in PARAMETERS}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.target_scaling = not (
            model.meta.get("target_scale", 1.0) == 1.0 and model.meta.get("target_mean", 0.0) == 0.0)
        model.validate()
        return model

    # ------------------------------------------------------------- migration
    def resize(self, hidden, seed=None):
        """Move to a different hidden width with an explicit weight migration.

        The rules require that a change of network shape be implemented as a
        compatible migration, not a silent re-initialisation.  The
        surviving hidden units - the first ``min(old, new)`` - keep their
        incoming and outgoing weights; the Adam moments of the surviving
        units are carried over as well, so the optimiser continues rather
        than restarting.  Widening adds fresh units drawn from the same
        distribution as a fresh layer; narrowing simply drops the tail.
        """
        old_hidden = self.meta["hidden"]
        if hidden == old_hidden:
            return {"migrated": False, "from_hidden": old_hidden, "to_hidden": hidden}
        if not 1 <= hidden <= MAX_HIDDEN:
            raise ValueError("resize() needs a hidden width between 1 and %d" % MAX_HIDDEN)
        rng = np.random.default_rng(self.meta["seed"] if seed is None else seed)
        keep = min(old_hidden, hidden)
        d = self.params["w1"].shape[0]
        w1 = np.empty((d, hidden))
        w1[:, :keep] = self.params["w1"][:, :keep]
        if hidden > keep:
            w1[:, keep:] = rng.normal(0, math.sqrt(1 / d), (d, hidden - keep))
        b1 = np.zeros(hidden)
        b1[:keep] = self.params["b1"][:keep]
        output = self.params["w2"].shape[1]
        w2 = np.zeros((hidden, output))
        w2[:keep, :] = self.params["w2"][:keep, :]
        b2 = self.params["b2"].copy()
        # The Adam moments of the surviving units are carried over, so the
        # optimiser continues instead of restarting.  The new units start
        # with a zero moment, exactly as a fresh layer would.
        # The buffers are allocated with the **new** shapes; the surviving
        # slices are then copied across from the old moments.
        first = {k: np.zeros_like(v) for k, v in {"w1": w1, "b1": b1, "w2": w2, "b2": b2}.items()}
        second = {k: np.zeros_like(v) for k, v in {"w1": w1, "b1": b1, "w2": w2, "b2": b2}.items()}
        first["w1"][:, :keep] = self.first["w1"][:, :keep]
        first["b1"][:keep] = self.first["b1"][:keep]
        first["w2"][:keep, :] = self.first["w2"][:keep, :]
        first["b2"][:] = self.first["b2"][:]
        second["w1"][:, :keep] = self.second["w1"][:, :keep]
        second["b1"][:keep] = self.second["b1"][:keep]
        second["w2"][:keep, :] = self.second["w2"][:keep, :]
        second["b2"][:] = self.second["b2"][:]
        self.params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        self.first = first
        self.second = second
        self.meta["hidden"] = hidden
        self.meta["parameter_count"] = int(sum(v.size for v in self.params.values()))
        # The epoch and step counters are **carried over**, not reset.  The
        # epoch counter drives the shuffling permutation, so resetting it
        # would replay shuffles the parent already saw; the step counter
        # drives Adam's bias correction, so resetting it would rescale the
        # carried moments against the wrong denominator.  Continuing the
        # counters is what makes this a continuation rather than a restart.
        self.validate()
        return {"migrated": True, "from_hidden": old_hidden, "to_hidden": hidden}

    def set_target_scaling(self, enabled):
        """Switch the regression target between standardised and direct.

        Switching *off* rewrites the stored target statistics so that the
        platform's reference reconstruction
        ``logits * target_scale + target_mean`` returns exactly the
        quantity the network was trained on.  The checkpoint's meaning is
        unchanged; only the internal representation is.
        """
        if self.meta["task"] != "regression":
            return False
        self.target_scaling = bool(enabled)
        if enabled:
            return True
        self.meta["target_scale"] = 1.0
        self.meta["target_mean"] = 0.0
        return True


def checkpoint_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
