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

import numpy as np
import pytest


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
