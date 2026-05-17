from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ForecastSplit:
    train_raw: pd.Series
    test_raw: pd.Series
    train_norm: pd.Series
    test_norm: pd.Series
    split_date: pd.Timestamp


@dataclass(frozen=True)
class LaggedTensors:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: np.ndarray


def make_lagged(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    values = series.values
    for i in range(len(series) - window):
        x_rows.append(values[i : i + window])
        y_rows.append(float(values[i + window]))
    return np.array(x_rows), np.array(y_rows)


def split_series(
    series: pd.Series,
    norm_series: pd.Series,
    *,
    window: int,
    prediction_length: int,
) -> ForecastSplit:
    split_date = series.index[-(window + prediction_length)]
    train_raw = series[series.index <= split_date]
    test_raw = series[series.index > split_date]
    train_norm = norm_series[norm_series.index <= split_date]
    test_norm = norm_series[norm_series.index > split_date]
    return ForecastSplit(
        train_raw=train_raw,
        test_raw=test_raw,
        train_norm=train_norm,
        test_norm=test_norm,
        split_date=split_date,
    )


def build_lagged_tensors(
    split: ForecastSplit,
    window: int,
) -> LaggedTensors:
    x_train, y_train = make_lagged(split.train_norm, window)
    x_test, y_test = make_lagged(split.test_norm, window)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    return LaggedTensors(
        x_train=x_train_t,
        y_train=y_train_t,
        x_test=x_test_t,
        y_test=y_test,
    )
