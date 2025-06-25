# multi_asset_backtester

A Python package for multi-asset backtesting. This project provides tools to fetch, store, and process financial data for backtesting strategies across multiple assets.

## Project Structure

- `data_fetcher/`: Contains modules for fetching and processing data.
- `data/`: Directory for raw CSV or Parquet files.
- `requirements.txt`: Python dependencies.
- `setup.py`: Makes this package pip-installable.

## Installation

```sh
pip install .
```

## Usage

Import and use the data fetcher in your scripts:

```python
from data_fetcher.fetcher import fetch_data
```

## License

MIT License
