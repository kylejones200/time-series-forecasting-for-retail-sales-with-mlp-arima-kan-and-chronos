from __future__ import annotations

import numpy as np
import pandas as pd

from retail_forecast.features import build_lagged_tensors, make_lagged, split_series


def test_make_lagged_shape() -> None:
    idx = pd.date_range("2020-01-01", periods=40, freq="MS")
    series = pd.Series(np.linspace(0, 1, len(idx)), index=idx)
    x, y = make_lagged(series, window=5)
    assert x.shape == (35, 5)
    assert y.shape == (35,)


def test_split_and_tensors() -> None:
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(0)
    raw = pd.Series(100 + rng.normal(0, 5, len(idx)).cumsum(), index=idx)
    norm = (raw - raw.mean()) / raw.std()
    split = split_series(raw, norm, window=12, prediction_length=6)
    tensors = build_lagged_tensors(split, window=12)
    assert tensors.x_train.ndim == 2
    assert tensors.y_train.shape[1] == 1
    assert len(tensors.y_test) > 0
