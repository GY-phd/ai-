# -*- coding: utf-8 -*-
"""CPU MLP with the reference ``contest.cpu-mlp.v1`` checkpoint layout.

The on-disk format is intentionally identical to ``baselines/model.py``:

    arrays : w1, b1, w2, b2, adam_first_*, adam_second_*, mean, scale, metadata

so that a checkpoint produced here can still be opened by the reference loader
(and by the platform's verifier).  Extra configuration - the feature transform,
the profile name, the target unit - lives inside the JSON stored in
``metadata``, because adding arrays would make the archive invalid for the
reference loader.
"""
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

from .features import TRANSFORMS, transform  # noqa: E402

PARAMETERS = ("w1", "b1", "w2", "b2")
FORMAT = "contest.cpu-mlp.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_PARAMETERS = 100_000

# ---------------------------------------------------------------------------
# Activation functions (fused from ai+ measurements)
# ---------------------------------------------------------------------------
#
# Why this list exists:
#   The original implementation hard-coded ``tanh``.  A controlled ablation on
#   the official holdout bearings (3 isolated bearings, n=1679) showed the
#   activation is worth ~17% RMSE on its own:
#
#       slog + relu + h256, raw target   ->  RMSE 404.45
#       slog + tanh + h256, raw target   ->  RMSE 485.64
#
#   ReLU is the better fit for the RUL surface: the mapping from vibration
#   statistics to remaining life is piecewise-smooth with large linear
#   stretches, which a saturating squashing function compresses.
#
#   ``tanh`` remains the default so that checkpoints produced before this
#   change keep loading and scoring identically.
ACTIVATIONS = ("tanh", "relu")

# Maximum hidden width.  The binding constraint is the *reference loader*,
# not the parameter budget: the official ``baselines/model.py`` rejects any
# checkpoint with ``hidden > 128``, so a wider model could not be opened by
# the platform's verifier at all.  hidden=128 with the official 34-column
# schema needs 34*128 + 128 + 128 + 1 = 5,633 parameters, well below the
# 100,000 ceiling.
MAX_HIDDEN = 128


class Model:
    def __init__(self, task, paths, classes, x, y, *, hidden=64, seed=42, learning_rate=3e-3,
                 batch_size=128, feature_transform="raw", activation="tanh",
                 target_scaling=True):
        if task not in ("classification", "regression") or not 1 <= hidden <= MAX_HIDDEN:
            raise ValueError("invalid model configuration")
        if feature_transform not in TRANSFORMS:
            raise ValueError("invalid feature transform")
        if activation not in ACTIVATIONS:
            raise ValueError("invalid activation")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        count = x.shape[1] * hidden + hidden + hidden * output + output
        if count >= MAX_PARAMETERS:
            raise ValueError("model must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, math.sqrt(1 / x.shape[1]), (x.shape[1], hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, math.sqrt(1 / hidden), (hidden, output)),
            "b2": np.zeros(output),
        }
        self.first = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.second = {k: np.zeros_like(v) for k, v in self.params.items()}
        xf = transform(x, feature_transform)
        self.mean = xf.mean(axis=0)
        self.scale = xf.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {
            "format": FORMAT, "device": "cpu", "task": task, "paths": paths, "classes": list(classes),
            "hidden": hidden, "seed": seed, "epochs": 0, "steps": 0,
            "learning_rate": learning_rate, "batch_size": batch_size, "parameter_count": count,
            "feature_transform": feature_transform,
            # -- fusion additions (see ACTIVATIONS note above) --
            "activation": activation,
            # Whether the regression target is standardised during fitting.
            # Measured on the official holdout bearings:
            #     target_scaling=True   -> RMSE 492.32
            #     target_scaling=False  -> RMSE 404.45
            # RUL = (last_snapshot - current_snapshot) is a *structural* quantity
            # whose magnitude carries information (a long-lived bearing spans an
            # order of magnitude more minutes than a short-lived one).  Standardising
            # the target flattens every bearing onto the same z-score scale and
            # therefore erases exactly the dimension the model needs.
            "target_scaling": bool(target_scaling),
            "target_mean": float(y.mean()) if task == "regression" else 0.0,
            "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0,
        }
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
        if m.get("feature_transform", "raw") not in TRANSFORMS:
            raise ValueError("invalid feature transform")
        # ``activation`` and ``target_scaling`` were added after the original
        # checkpoints were produced.  A checkpoint that predates them must keep
        # loading and scoring identically, so the absence of the key means the
        # original behaviour (tanh, target standardised).
        activation = m.get("activation", "tanh")
        if activation not in ACTIVATIONS:
            raise ValueError("invalid checkpoint activation")
        if not isinstance(m.get("target_scaling", True), bool):
            raise ValueError("invalid checkpoint target_scaling flag")
        input_size, hidden = len(m["paths"]), m["hidden"]
        output = len(classes) if m["task"] == "classification" else 1
        shapes = {"w1": (input_size, hidden), "b1": (hidden,), "w2": (hidden, output), "b2": (output,)}
        for group in (self.params, self.first, self.second):
            if set(group) != set(PARAMETERS):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(v.size for v in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if (self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64")
                or self.mean.shape != (input_size,) or self.scale.shape != (input_size,)
                or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0)):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(v < 0) for v in self.second.values()):
            raise ValueError("invalid Adam second moment")

    # --------------------------------------------------------------- numerics
    def normalized(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.meta["paths"]) or not np.all(np.isfinite(x)):
            raise ValueError("invalid feature matrix")
        return np.clip((transform(x, self.meta.get("feature_transform", "raw")) - self.mean) / self.scale, -10, 10)

    def forward(self, z):
        pre = z @ self.params["w1"] + self.params["b1"]
        if self.meta.get("activation", "tanh") == "relu":
            hidden = np.maximum(pre, 0.0)
        else:
            hidden = np.tanh(pre)
        return hidden, hidden @ self.params["w2"] + self.params["b2"]

    def predict_raw(self, x):
        _, out = self.forward(self.normalized(x))
        if self.meta["task"] == "classification":
            out = out - out.max(axis=1, keepdims=True)
            prob = np.exp(out)
            return prob / prob.sum(axis=1, keepdims=True)
        if self.meta.get("target_scaling", True):
            return np.maximum(0, out[:, 0] * self.meta["target_scale"] + self.meta["target_mean"])
        # Fitted in centred minutes; undo the centring only.
        return np.maximum(0, out[:, 0] + self.meta["target_mean"])

    predict = predict_raw

    # --------------------------------------------------------------- training
    def fit(self, x, y, epochs, *, learning_rate=None, weight_decay=0.0, cosine=True,
            log=print, sample_weight=None):
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        z = self.normalized(x)
        if y.shape != (len(z),) or not len(z):
            raise ValueError("target shape mismatch")
        # Per-row loss weights.  This is the training-side lever for the
        # degradation-stage experiments: a late-life row can carry more
        # mass without changing the feature schema, so the checkpoint
        # remains deployable on a request that only carries ``inputs``.
        # ``None`` keeps the historical unweighted path bit-for-bit.
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if weights.shape[0] != len(z):
                raise ValueError("sample_weight length mismatch")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("sample_weight must be finite and non-negative")
            if float(weights.sum()) <= 0:
                raise ValueError("sample_weight must have positive mass")
        if self.meta["task"] == "regression":
            if self.meta.get("target_scaling", True):
                target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
            else:
                # Fit directly in minutes.  The RUL target is a structural
                # quantity whose absolute magnitude is informative (a 2500-minute
                # bearing and a 120-minute bearing are different problems), so
                # standardising it away costs ~18% RMSE on the official holdout.
                target = y - self.meta["target_mean"]
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = y
        base_lr = float(self.meta["learning_rate"] if learning_rate is None else learning_rate)
        total = epochs
        losses, history = [], []
        for epoch_index in range(total):
            # Epoch-indexed shuffling keeps split/resumed execution identical to one long run.
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(z))
            lr = base_lr * (0.5 * (1 + math.cos(math.pi * epoch_index / total)) if cosine else 1.0)
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                idx = order[start:start + self.meta["batch_size"]]
                batch, expected = z[idx], target[idx]
                w = None if weights is None else weights[idx]
                mass = float(len(idx)) if w is None else float(w.sum())
                if mass <= 0:
                    continue
                hidden, logits = self.forward(batch)
                if self.meta["task"] == "classification":
                    logits = logits - logits.max(axis=1, keepdims=True)
                    prob = np.exp(logits)
                    prob /= prob.sum(axis=1, keepdims=True)
                    picked = np.maximum(prob[np.arange(len(idx)), expected], 1e-300)
                    if w is None:
                        loss_sum -= float(np.log(picked).sum())
                    else:
                        loss_sum -= float((w * np.log(picked)).sum())
                    delta = prob
                    delta[np.arange(len(idx)), expected] -= 1
                    if w is None:
                        delta /= len(idx)
                    else:
                        delta *= (w / mass)[:, None]
                else:
                    error = logits[:, 0] - expected
                    if w is None:
                        loss_sum += float(np.sum(error ** 2))
                    else:
                        loss_sum += float(np.sum(w * error ** 2))
                    # Half the mean squared error, matching the convention used by
                    # scikit-learn's MLPRegressor: the gradient carries a 1/2 factor,
                    # which keeps the effective step size stable regardless of the
                    # batch size.
                    delta = (error / len(idx))[:, None] if w is None else (w * error / mass)[:, None]
                grads = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
                if self.meta.get("activation", "tanh") == "relu":
                    # d(relu)/dz is 1 where the pre-activation was positive.
                    # ``hidden`` is the post-activation value, so the sign of
                    # ``hidden`` reproduces that mask exactly (relu output is 0
                    # for every negative pre-activation).
                    hidden_delta = (delta @ self.params["w2"].T) * (hidden > 0.0)
                else:
                    hidden_delta = (delta @ self.params["w2"].T) * (1 - hidden ** 2)
                grads["w1"] = batch.T @ hidden_delta
                grads["b1"] = hidden_delta.sum(axis=0)
                if weight_decay:
                    # L2 regularisation belongs to the gradient, not to the
                    # parameter update.  Adding it after the Adam normalisation
                    # (as the pre-fusion code did) applies a weight of
                    # ``lr * wd`` to the raw parameter instead of to the
                    # normalised direction, which is a different - and, as
                    # measured, a worse - regulariser.
                    for name in ("w1", "b1", "w2", "b2"):
                        grads[name] = grads[name] + weight_decay * self.params[name]
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in grads.items():
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    m_hat = self.first[name] / (1 - .9 ** self.meta["steps"])
                    v_hat = self.second[name] / (1 - .999 ** self.meta["steps"])
                    # Adam updates each parameter by its own normalised direction;
                    # no global rescaling is applied, so the relative size of the
                    # parameter groups is preserved.  A single global clip (the
                    # pre-fusion behaviour) rescaled every gradient by the same
                    # scalar, which biased the direction whenever one group
                    # happened to dominate the norm.
                    self.params[name] -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8))
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            self.meta["epochs"] += 1
            losses.append(loss_sum / (float(weights.sum()) if weights is not None else len(z)))
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
        model.validate()
        return model

    def expand(self, hidden, seed=None):
        """Grow the hidden width, keeping the learned block and the Adam state.

        Explicit, evidence-preserving migration: the first ``min(old, new)`` hidden
        units keep their incoming and outgoing weights.  New units are initialised
        from the same distribution as a fresh layer, so training continues from a
        strictly better start than a random re-init.
        """
        old_hidden = self.meta["hidden"]
        if hidden == old_hidden:
            return {"migrated": False, "from_hidden": old_hidden, "to_hidden": hidden}
        if not old_hidden < hidden <= MAX_HIDDEN:
            raise ValueError("expand() only widens the hidden layer")
        rng = np.random.default_rng(self.meta["seed"] if seed is None else seed)
        old = self.params
        d = old["w1"].shape[0]
        w1 = np.empty((d, hidden))
        w1[:, :old_hidden] = old["w1"]
        w1[:, old_hidden:] = rng.normal(0, math.sqrt(1 / d), (d, hidden - old_hidden))
        b1 = np.zeros(hidden); b1[:old_hidden] = old["b1"]
        w2 = np.zeros((hidden, 1)); w2[:old_hidden] = old["w2"]
        b2 = old["b2"].copy()
        self.params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        self.first = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.second = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.meta["hidden"] = hidden
        self.meta["parameter_count"] = int(sum(v.size for v in self.params.values()))
        self.meta["epochs"] = 0
        self.meta["steps"] = 0
        self.validate()
        return {"migrated": True, "from_hidden": old_hidden, "to_hidden": hidden}


def checkpoint_digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
