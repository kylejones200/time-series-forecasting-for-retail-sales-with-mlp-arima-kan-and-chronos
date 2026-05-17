# Time Series Forecasting for Retail Sales with MLP, ARIMA, KAN, and Chronos

Published: 2025-04-22  
Medium: [Time Series Forecasting for Retail Sales with MLP, ARIMA, KAN, and Chronos](https://medium.com/@kyle-t-jones/time-series-forecasting-for-retail-sales-with-mlp-arima-kan-and-chronos-258abbbf4779)

Benchmarks ARIMA, MLP, KAN, LSTM, and Amazon Chronos on FRED retail sales (`RSAFS`). Companion code for the article (`article.md`).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run retail-forecast-run --no-chronos   # skip large HF model on first run
uv run retail-forecast-run                # full benchmark including Chronos
```

Outputs:

| Path | Contents |
|------|----------|
| `outputs/figures/` | Forecast comparison chart |
| `outputs/results.json` | RMSE, MAE, MAPE, and train times per model |

## Project layout

```
config.yaml              # FRED series, horizons, model hyperparameters
config.local.yaml.example
pyproject.toml / uv.lock
src/retail_forecast/     # data loading, models, metrics, plots, CLI
scripts/cache_fred.py    # download FRED series to data/ for offline use
notebooks/               # original exploratory notebooks
legacy/                  # article export scripts (reference)
data/                    # optional cached CSV (see data/README.md)
outputs/figures/         # generated plots
tests/
article.md
```

## Configuration

Edit `config.yaml`:

- `data.source` — `fred` (default) or `csv`
- `data.series_id` — FRED code (default `RSAFS`, Advance Retail Sales)
- `forecast.prediction_length` / `forecast.window` — holdout horizon and lag window
- `models.chronos.enabled` — set `false` to skip Chronos
- `models.*.epochs` — training iterations for torch/KAN models

Machine-specific overrides: copy `config.local.yaml.example` to `config.local.yaml` (gitignored).

## Data

See [data/README.md](data/README.md). To cache FRED data locally:

```bash
uv run python scripts/cache_fred.py
```

Then set `data.source: csv` in `config.local.yaml`.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests scripts
```

Optional serif styling (matches the article plots):

```bash
uv sync --extra plot
```

CI runs ruff and pytest with Chronos disabled and minimal training epochs.

## License

MIT — see [LICENSE](LICENSE).

## Notebooks

Exploratory notebooks live in `notebooks/`. Use the project venv (`uv sync`, then launch Jupyter from the repo root) so imports resolve to `retail_forecast`.
