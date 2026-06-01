from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_forecast.runner import run


@pytest.fixture
def fast_config(tmp_path: Path) -> dict:
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(1)
    values = 400_000 + rng.normal(0, 1000, len(idx)).cumsum()
    csv_path = tmp_path / "series.csv"
    pd.DataFrame({"date": idx, "value": values}).to_csv(csv_path, index=False)
    return {
        "logging": {"level": "WARNING"},
        "data": {
            "source": "csv",
            "csv_path": str(csv_path),
            "seed": 0,
        },
        "forecast": {"prediction_length": 6, "window": 12},
        "models": {
            "arima": {"order": [1, 1, 0]},
            "mlp": {"epochs": 2, "learning_rate": 0.05},
            "kan": {"epochs": 2, "learning_rate": 0.05},
            "lstm": {"epochs": 2, "learning_rate": 0.05},
            "chronos": {"enabled": False},
        },
        "output": {
            "figures_dir": str(tmp_path / "figures"),
            "results_path": str(tmp_path / "results.json"),
            "save_figures": True,
        },
    }


def test_run_pipeline(fast_config: dict) -> None:
    result = run(cfg=fast_config, include_chronos=False)
    assert "ARIMA" in result["results"].index
    assert "MLP" in result["results"].index
    assert result["results_path"].is_file()
    payload = json.loads(result["results_path"].read_text())
    assert len(payload["metrics"]) >= 4
    assert result["figure_path"] is not None
    assert result["figure_path"].is_file()
