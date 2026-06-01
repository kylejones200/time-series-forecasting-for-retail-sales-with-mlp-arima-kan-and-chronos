from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from retail_forecast.models import ModelPrediction


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs((y - yhat) / (y + 1e-10))) * 100)


def evaluate(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float, float]:
    return (
        mean_squared_error(y, yhat) / 1e6,
        mean_absolute_error(y, yhat) / 1e6,
        mape(y, yhat),
    )


def results_dataframe(
    y_actual: np.ndarray,
    predictions: list[ModelPrediction],
) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        rmse, mae, mape_pct = evaluate(y_actual, pred.values)
        rows.append(
            {
                "Model": pred.name,
                "RMSE (M USD)": rmse,
                "MAE (M USD)": mae,
                "MAPE (%)": mape_pct,
                "Train Time (s)": pred.train_seconds,
            }
        )
    return pd.DataFrame(rows).set_index("Model")
