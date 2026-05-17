from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter

from retail_forecast.models import ModelPrediction
from retail_forecast.paths import resolve_project_path


def _apply_plot_style(cfg: dict[str, Any]) -> None:
    out_cfg = cfg.get("output") or {}
    font = out_cfg.get("font_family")
    if font:
        try:
            import signalplot

            signalplot.apply(font_family=font)
        except ImportError:
            plt.rcParams["font.family"] = font


def save_forecast_comparison(
    series: pd.Series,
    predictions: list[ModelPrediction],
    *,
    prediction_length: int,
    cfg: dict[str, Any],
) -> Path:
    out_cfg = cfg.get("output") or {}
    _apply_plot_style(cfg)

    zoom_months = int(out_cfg.get("zoom_months", 24))
    prediction_start = series.index[-prediction_length]
    zoom_start = series.index[-min(zoom_months, len(series))]
    series_zoom = series[series.index >= zoom_start]

    valid_length = min(
        prediction_length,
        *(len(p.values) for p in predictions),
    )
    forecast_index = series.index[series.index >= prediction_start][:valid_length]
    end_date = forecast_index[-1]

    fig, ax = plt.subplots(figsize=tuple(out_cfg.get("figsize", [12, 5])))
    ax.plot(
        series_zoom.index,
        series_zoom.values,
        label="Actual",
        color="black",
        linewidth=2,
    )
    ax.axvline(prediction_start, color="lightgray", linestyle="--", linewidth=1)

    styles = {
        "ARIMA": {"linestyle": "--"},
        "MLP": {},
        "KAN": {"linestyle": "-."},
        "LSTM": {"linestyle": ":"},
        "Chronos T5": {"linestyle": "dotted"},
    }
    for pred in predictions:
        values = pred.values[:valid_length]
        ax.plot(
            forecast_index,
            values,
            linewidth=1.5,
            **styles.get(pred.name, {}),
        )
        ax.text(
            end_date + pd.DateOffset(months=1),
            values[-1],
            pred.name,
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.set_ylabel("Sales (Millions USD)")
    ax.set_title("Retail Sales Forecasts (zoomed; grey line = forecast start)")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.tight_layout()

    figures_dir = resolve_project_path(out_cfg.get("figures_dir", "outputs/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    chart_name = out_cfg.get("forecast_chart", "retail_forecast_comparison.png")
    out_path = figures_dir / chart_name
    dpi = int(out_cfg.get("figure_dpi", 120))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
