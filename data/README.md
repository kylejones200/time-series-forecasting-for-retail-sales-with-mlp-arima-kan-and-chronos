# Data

By default the pipeline loads **RSAFS** (Advance Retail Sales) from FRED via `pandas-datareader`.

To work offline, cache a CSV:

```bash
uv run python scripts/cache_fred.py
```

This writes `data/rsafs.csv`. Then in `config.local.yaml`:

```yaml
data:
  source: csv
  csv_path: data/rsafs.csv
```

The CSV must have `date` and `value` columns (configurable via `date_column` / `value_column` in `config.yaml`).
