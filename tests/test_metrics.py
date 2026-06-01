from __future__ import annotations

import numpy as np

from retail_forecast.metrics import evaluate, mape


def test_mape_and_evaluate() -> None:
    y = np.array([100.0, 110.0, 120.0])
    yhat = np.array([101.0, 108.0, 122.0])
    assert mape(y, yhat) < 5.0
    rmse, mae, mape_pct = evaluate(y, yhat)
    assert rmse >= 0
    assert mae >= 0
    assert mape_pct < 5.0
