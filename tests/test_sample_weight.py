"""Regression test: sample_weight routing for pipeline-wrapped estimators.

Regression context:
    When `lgbm` was the base model, `Trainer.train` passed
    `sample_weight=` straight to `fit()`. That worked because lgbm is a bare
    estimator.

    After switching the RUL base to `mlp`, the estimator is wrapped as
        TransformedTargetRegressor(Pipeline([StandardScaler, MLPRegressor]))
    and sklearn's Pipeline rejects a bare `sample_weight` fit kwarg:
        ValueError: Pipeline.fit does not accept the sample_weight parameter

    This test locks in the routing behaviour so switching the base model
    again cannot silently break the retraining path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ``evo`` is a source tree, not an installed package.  ``test_core`` adds
# this path during collection, but a single-file run of this module does
# not go through ``test_core``; add it here so the file is self-contained.
_EVO_SRC = Path(__file__).resolve().parent.parent / "evo" / "src"
if _EVO_SRC.is_dir() and str(_EVO_SRC) not in sys.path:
    sys.path.insert(0, str(_EVO_SRC))


def _tiny_data(n: int = 120, d: int = 6):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize("name", ["mlp", "lgbm", "logreg"])
def test_train_accepts_sample_weight(name):
    """Trainer.train must accept sample_weight for every regression model."""
    from evo.config import load_config
    from evo.models import Trainer

    cfg = load_config()
    cfg.set("data.task", "regression")
    cfg.set("model.mlp_params.max_iter", 30)
    cfg.set("model.mlp_params.hidden_layer_sizes", [16])
    cfg.set("model.lgbm_params.n_estimators", 10)

    X, y = _tiny_data()
    w = np.linspace(0.5, 2.0, len(y))

    trainer = Trainer(cfg)
    bundle = trainer.train(
        name,
        X,
        y,
        [f"f{i}" for i in range(X.shape[1])],
        sample_weight=w,
    )
    assert bundle.estimator is not None
    pred = bundle.estimator.predict(X)
    assert np.all(np.isfinite(pred))


def test_weight_fit_kwargs_routes_to_pipeline_step():
    """The routing helper must target the last pipeline step."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from evo.models import Trainer

    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    kw = Trainer._weight_fit_kwargs(pipe, np.ones(3))
    assert "model__sample_weight" in kw
    assert "sample_weight" not in kw

    # Bare estimator: plain sample_weight.
    kw = Trainer._weight_fit_kwargs(Ridge(), np.ones(3))
    assert "sample_weight" in kw


def test_weight_fit_kwargs_routes_through_target_regressor():
    """TransformedTargetRegressor must not receive the weight.

    The installed sklearn version has a routing bug for
    TransformedTargetRegressor: passing any `regressor__*` fit kwarg raises
    `KeyError: 'regressor'`. Since `model.target_scaling` is off by default
    (target scaling measurably HURTS the RUL task), the safe behaviour is to
    drop the weight rather than crash the whole retraining.
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from evo.models import Trainer

    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    ttr = TransformedTargetRegressor(regressor=pipe)
    kw = Trainer._weight_fit_kwargs(ttr, np.ones(3))
    assert kw == {}


def test_target_scaling_is_off_by_default():
    """Regression target scaling must stay off (it hurt RUL accuracy).

    Measured on the official holdout bearings:
        TransformedTargetRegressor -> RMSE 716.03
        input-scaling only         -> RMSE 521.07
    """
    from evo.config import load_config
    from evo.models import ModelFactory

    cfg = load_config()
    cfg.set("data.task", "regression")
    assert not bool(cfg.get("model.target_scaling", False))

    factory = ModelFactory(cfg)
    est = factory.build("mlp")
    assert est.__class__.__name__ == "Pipeline"
    assert [name for name, _ in est.steps] == ["scaler", "model"]


def test_rulkit_model_unweighted_path_is_bit_identical():
    """Adding the weighted path must not change the historical unweighted fit."""
    from rulkit.model import Model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(320, 6))
    y = x[:, 0] * 2.0 - x[:, 1] + 0.2 * rng.normal(size=320)

    def fit(weights):
        model = Model("regression", [f"f{i}" for i in range(6)], [], x, y,
                      hidden=16, seed=7, learning_rate=0.01, batch_size=32,
                      feature_transform="raw", target_scaling=False)
        losses = model.fit(x, y, 25, sample_weight=weights)
        return model, losses

    plain, plain_losses = fit(None)
    ones, one_losses = fit(np.ones(len(y)))
    assert np.array_equal(plain_losses, one_losses)
    assert all(np.array_equal(plain.params[k], ones.params[k]) for k in plain.params)


def test_rulkit_model_accepts_degradation_weights():
    """Late-life weights must run and produce a finite objective."""
    from rulkit.model import Model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(320, 6))
    y = np.abs(x[:, 0] * 2.0 - x[:, 1]) + 1.0
    weights = 1.0 + 1.0 * (1.0 - y / y.max())
    model = Model("regression", [f"f{i}" for i in range(6)], [], x, y,
                  hidden=16, seed=7, learning_rate=0.01, batch_size=32,
                  feature_transform="raw", target_scaling=False)
    losses = model.fit(x, y, 25, sample_weight=weights)
    assert np.all(np.isfinite(losses))
    assert np.all(np.isfinite(model.predict(x)))
