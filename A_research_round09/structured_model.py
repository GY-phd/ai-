"""Portable copy of the round-08 fixed feature map and structured head."""
from pathlib import Path
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np

def softmax(scores):
    values = np.exp(scores - scores.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)

def sensor_matrix(x):
    if x.ndim != 2 or x.shape[1] != 128 or not np.isfinite(x).all():
        raise ValueError("expected finite N x 128 features")
    return x.reshape(len(x), 16, 8)

def shape_features(x):
    sensors = sensor_matrix(x)
    response = sensors[:, :, [0, 2, 3, 4, 5, 6, 7]]
    magnitude = np.maximum(np.sqrt(np.mean(response ** 2, axis=2, keepdims=True)), 1e-12)
    return np.arcsinh(response / magnitude).reshape(len(x), 112)

class StructuredModel:
    def features(self, x):
        shape = shape_features(x)
        if self.variant == "shape":
            return shape
        sensors = sensor_matrix(x)
        amplitude = np.stack((sensors[:, :, 0], sensors[:, :, 1] - 1), axis=2).reshape(len(x), 32)
        return np.column_stack((shape, np.arcsinh(amplitude / self.amplitude_scale)))

    def design(self, x):
        scaled = np.clip((self.features(x) - self.center) / self.scale, -6, 6)
        return np.column_stack((np.ones(len(x)), scaled))

    @classmethod
    def fit(cls, x, y, kind="ridge", variant="hybrid"):
        if kind not in ("ridge", "lda") or variant not in ("shape", "hybrid", "equal", "empirical"):
            raise ValueError("unknown fixed configuration")
        model = cls()
        model.kind, model.variant = kind, variant
        sensors = sensor_matrix(x)
        amplitude = np.stack((sensors[:, :, 0], sensors[:, :, 1] - 1), axis=2).reshape(len(x), 32)
        model.amplitude_scale = np.maximum(np.median(np.abs(amplitude), axis=0), 1e-12)
        values = model.features(x)
        model.center = np.median(values, axis=0)
        low, high = np.percentile(values, (25, 75), axis=0)
        model.scale = np.maximum(high - low, max(1e-6, .1 * float(np.median(high - low))))
        design = model.design(x)
        counts = np.bincount(y, minlength=6)
        if np.any(counts == 0):
            raise ValueError("source training must contain all six classes")
        if kind == "ridge":
            weights = len(y) / (6 * counts[y])
            model.coef = np.linalg.solve(design.T @ (design * weights[:, None]) + np.eye(design.shape[1]),
                                         design.T @ (np.eye(6)[y] * weights[:, None]))
        else:
            z = design[:, 1:]
            centers = np.stack([z[y == label].mean(axis=0) for label in range(6)])
            residual = z - centers[y]
            covariance = residual.T @ residual / (len(y) - 6)
            dimension = z.shape[1]
            covariance = .2 * covariance + .8 * np.eye(dimension) * np.trace(covariance) / dimension
            coefficients = np.linalg.solve(covariance + 1e-6 * np.eye(dimension), centers.T)
            prior = counts / len(y) if variant == "empirical" else np.full(6, 1 / 6)
            intercept = -.5 * (centers.T * coefficients).sum(axis=0) + np.log(prior)
            model.coef = np.vstack((intercept, coefficients))
        return model

    def predict(self, x):
        # Scores support argmax classification. Calibration is not established.
        return softmax(self.design(x) @ self.coef)

    def save(self, path):
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, format="round08.structured.v1", kind=self.kind,
                                variant=self.variant, amplitude_scale=self.amplitude_scale,
                                center=self.center, scale=self.scale, coef=self.coef)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as values:
            if str(values["format"]) != "round08.structured.v1":
                raise ValueError("unknown checkpoint format")
            model.kind, model.variant = str(values["kind"]), str(values["variant"])
            for key in ("amplitude_scale", "center", "scale", "coef"):
                setattr(model, key, values[key].copy())
        expected = 112 if model.variant == "shape" else 144
        if (model.center.shape != (expected,) or model.scale.shape != (expected,)
                or model.coef.shape != (expected + 1, 6) or model.amplitude_scale.shape != (32,)
                or any(not np.isfinite(getattr(model, k)).all() for k in ("amplitude_scale", "center", "scale", "coef"))
                or np.any(model.scale <= 0) or np.any(model.amplitude_scale <= 0)):
            raise ValueError("invalid checkpoint state")
        return model
