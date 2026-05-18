from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

from retail_forecast.features import LaggedTensors


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int = 1, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


@dataclass(frozen=True)
class ModelPrediction:
    name: str
    values: np.ndarray
    train_seconds: float


def _train_torch(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def fit_arima(
    train_raw: pd.Series,
    test_raw: pd.Series,
    *,
    order: tuple[int, int, int],
    prediction_length: int,
) -> ModelPrediction:
    start = time.perf_counter()
    model = ARIMA(train_raw, order=order).fit()
    forecast = model.forecast(steps=prediction_length)
    forecast.index = test_raw.index[:prediction_length]
    elapsed = time.perf_counter() - start
    return ModelPrediction("ARIMA", forecast.values, elapsed)


def fit_mlp(
    tensors: LaggedTensors,
    *,
    window: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    dropout: float,
) -> ModelPrediction:
    model = MLP(window, hidden=hidden_size, dropout=dropout)
    elapsed = _train_torch(
        model,
        tensors.x_train,
        tensors.y_train,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    model.eval()
    with torch.no_grad():
        pred = model(tensors.x_test).squeeze().numpy()
    return ModelPrediction("MLP", pred, elapsed)


def fit_kan(
    tensors: LaggedTensors,
    *,
    window: int,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
    grid: int,
    k: int,
) -> ModelPrediction:
    from kan import KAN

    model = KAN([window, hidden_size, 1], grid=grid, k=k)
    elapsed = _train_torch(
        model,
        tensors.x_train,
        tensors.y_train,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    model.eval()
    with torch.no_grad():
        pred = model(tensors.x_test).squeeze().numpy()
    return ModelPrediction("KAN", pred, elapsed)


def fit_lstm(
    tensors: LaggedTensors,
    *,
    epochs: int,
    learning_rate: float,
    hidden_size: int,
) -> ModelPrediction:
    x_train = tensors.x_train.unsqueeze(-1)
    x_test = tensors.x_test.unsqueeze(-1)
    model = LSTMForecaster(input_dim=1, hidden=hidden_size)
    elapsed = _train_torch(
        model,
        x_train,
        tensors.y_train,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    model.eval()
    with torch.no_grad():
        pred = model(x_test).squeeze().numpy()
    return ModelPrediction("LSTM", pred, elapsed)


def fit_chronos(
    train_raw: pd.Series,
    *,
    prediction_length: int,
    model_id: str,
) -> ModelPrediction:
    from chronos import ChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    start = time.perf_counter()
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )
    context = torch.tensor(train_raw.values, dtype=torch.float32)
    forecast = pipeline.predict(context, prediction_length=prediction_length)
    pred = forecast[0].median(dim=0).values.numpy()
    elapsed = time.perf_counter() - start
    return ModelPrediction("Chronos T5", pred, elapsed)


def denormalize_predictions(
    predictions: list[ModelPrediction],
    *,
    std_val: float,
    mean_val: float,
    y_test_norm: np.ndarray,
    prediction_length: int,
) -> tuple[np.ndarray, list[ModelPrediction]]:
    y_actual = y_test_norm * std_val + mean_val
    valid_length = min(len(y_actual), prediction_length)
    y_actual = y_actual[-valid_length:]
    denormed: list[ModelPrediction] = []
    for pred in predictions:
        values = pred.values
        if pred.name == "ARIMA":
            trimmed = values[:valid_length]
        else:
            trimmed = values[-valid_length:] * std_val + mean_val
        denormed.append(ModelPrediction(pred.name, trimmed, pred.train_seconds))

    return y_actual, denormed


def model_cfg(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    return (cfg.get("models") or {}).get(key) or {}
