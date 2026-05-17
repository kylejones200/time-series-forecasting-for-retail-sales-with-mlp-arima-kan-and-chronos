from __future__ import annotations

import datetime
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from retail_forecast.paths import resolve_project_path


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_series(cfg: dict[str, Any]) -> pd.Series:
    """Load a monthly FRED series or a local CSV for offline use."""
    data_cfg = cfg.get("data") or {}
    source = str(data_cfg.get("source", "fred")).lower()

    if source == "csv":
        csv_path = resolve_project_path(data_cfg.get("csv_path", "data/rsafs.csv"))
        date_col = str(data_cfg.get("date_column", "date"))
        value_col = str(data_cfg.get("value_column", "value"))
        frame = pd.read_csv(csv_path, parse_dates=[date_col])
        series = frame.set_index(date_col)[value_col].astype(float)
    elif source == "fred":
        from pandas_datareader import data as web

        series_id = str(data_cfg.get("series_id", "RSAFS"))
        start = pd.Timestamp(data_cfg.get("fred_start", "2000-01-01"))
        end = pd.Timestamp(data_cfg.get("fred_end", datetime.date.today().isoformat()))
        frame = web.DataReader(series_id, "fred", start, end).dropna()
        series = frame[series_id]
    else:
        raise ValueError(f"Unknown data.source: {source!r} (use 'fred' or 'csv')")

    series = series.sort_index()
    series.index = pd.DatetimeIndex(series.index)
    series.index.freq = "MS"
    return series


def normalize_series(series: pd.Series) -> tuple[pd.Series, float, float]:
    mean_val = float(series.mean())
    std_val = float(series.std())
    if std_val == 0:
        raise ValueError("Cannot normalize a constant series (std=0)")
    norm = (series - mean_val) / std_val
    return norm, mean_val, std_val
