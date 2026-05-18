from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from retail_forecast import __version__
from retail_forecast.config import configure_logging, load_config
from retail_forecast.data import load_series, normalize_series, set_seeds
from retail_forecast.features import build_lagged_tensors, split_series
from retail_forecast.metrics import results_dataframe
from retail_forecast.models import (
    ModelPrediction,
    denormalize_predictions,
    fit_arima,
    fit_chronos,
    fit_kan,
    fit_lstm,
    fit_mlp,
    model_cfg,
)
from retail_forecast.paths import DEFAULT_CONFIG_PATH, path_relative_to_project
from retail_forecast.plots import save_forecast_comparison

logger = logging.getLogger(__name__)


def run(
    config_path: Path | str | None = None,
    *,
    cfg: dict[str, Any] | None = None,
    include_chronos: bool | None = None,
) -> dict[str, Any]:
    if cfg is None:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        cfg = load_config(path)
    configure_logging(cfg)
    data_cfg = cfg.get("data") or {}
    seed = int(data_cfg.get("seed", 42))
    set_seeds(seed)
    forecast_cfg = cfg.get("forecast") or {}
    prediction_length = int(forecast_cfg.get("prediction_length", 12))
    window = int(forecast_cfg.get("window", 24))
    series = load_series(cfg)
    norm_series, mean_val, std_val = normalize_series(series)
    split = split_series(series, norm_series, window=window, prediction_length=prediction_length)
    tensors = build_lagged_tensors(split, window)
    arima_cfg = model_cfg(cfg, "arima")
    order_list = arima_cfg.get("order", [5, 1, 0])
    order = tuple(int(x) for x in order_list)
    predictions: list[ModelPrediction] = [
        fit_arima(
            split.train_raw,
            split.test_raw,
            order=order,
            prediction_length=prediction_length,
        )
    ]
    mlp_cfg = model_cfg(cfg, "mlp")
    predictions.append(
        fit_mlp(
            tensors,
            window=window,
            epochs=int(mlp_cfg.get("epochs", 200)),
            learning_rate=float(mlp_cfg.get("learning_rate", 0.01)),
            hidden_size=int(mlp_cfg.get("hidden_size", 64)),
            dropout=float(mlp_cfg.get("dropout", 0.2)),
        )
    )
    kan_cfg = model_cfg(cfg, "kan")
    predictions.append(
        fit_kan(
            tensors,
            window=window,
            epochs=int(kan_cfg.get("epochs", 200)),
            learning_rate=float(kan_cfg.get("learning_rate", 0.01)),
            hidden_size=int(kan_cfg.get("hidden_size", 32)),
            grid=int(kan_cfg.get("grid", 3)),
            k=int(kan_cfg.get("k", 2)),
        )
    )
    lstm_cfg = model_cfg(cfg, "lstm")
    predictions.append(
        fit_lstm(
            tensors,
            epochs=int(lstm_cfg.get("epochs", 200)),
            learning_rate=float(lstm_cfg.get("learning_rate", 0.01)),
            hidden_size=int(lstm_cfg.get("hidden_size", 32)),
        )
    )
    chronos_cfg = model_cfg(cfg, "chronos")
    use_chronos = (
        bool(chronos_cfg.get("enabled", True)) if include_chronos is None else include_chronos
    )
    if use_chronos:
        predictions.append(
            fit_chronos(
                split.train_raw,
                prediction_length=prediction_length,
                model_id=str(chronos_cfg.get("model_id", "amazon/chronos-t5-large")),
            )
        )

    y_actual, predictions = denormalize_predictions(
        predictions,
        std_val=std_val,
        mean_val=mean_val,
        y_test_norm=tensors.y_test,
        prediction_length=prediction_length,
    )
    results = results_dataframe(y_actual, predictions)
    logger.info("\n%s", results.to_string())
    out_cfg = cfg.get("output") or {}
    figure_path: Path | None = None
    if out_cfg.get("save_figures", True):
        figure_path = save_forecast_comparison(
            series,
            predictions,
            prediction_length=prediction_length,
            cfg=cfg,
        )
        logger.info("Wrote %s", figure_path)

    results_path = Path(out_cfg.get("results_path", "outputs/results.json"))
    if not results_path.is_absolute():
        from retail_forecast.paths import resolve_project_path

        results_path = resolve_project_path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        "series_id": str(data_cfg.get("series_id", "RSAFS")),
        "split_date": split.split_date.isoformat(),
        "metrics": results.reset_index().to_dict(orient="records"),
        "figure": path_relative_to_project(figure_path) if figure_path else None,
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", results_path)
    return {
        "series": series,
        "results": results,
        "predictions": predictions,
        "figure_path": figure_path,
        "results_path": results_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retail sales forecasting benchmark (ARIMA, MLP, KAN, LSTM, Chronos)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--no-chronos",
        action="store_true",
        help="Skip Chronos (faster; no Hugging Face download)",
    )
    args = parser.parse_args()
    run(args.config, include_chronos=not args.no_chronos)


if __name__ == "__main__":
    main()
