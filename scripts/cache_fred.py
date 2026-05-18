"""Download the configured FRED series to data/ for offline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from retail_forecast.config import load_config
from retail_forecast.data import load_series
from retail_forecast.paths import DEFAULT_CONFIG_PATH, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache FRED series to CSV")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    series = load_series(cfg)
    data_cfg = cfg.get("data") or {}
    out_path = resolve_project_path(data_cfg.get("csv_path", "data/rsafs.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.rename("value").to_frame()
    frame.index.name = "date"
    frame.reset_index().to_csv(out_path, index=False)
    print(f"Wrote {len(frame)} rows to {out_path}")


if __name__ == "__main__":
    main()
