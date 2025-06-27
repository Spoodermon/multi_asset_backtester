# Quantitative Multi-Asset Backtesting Framework

This repository contains an event-driven backtesting framework in Python, engineered to develop and validate quantitative trading strategies using advanced machine learning techniques. The entire system is built as a modular and installable Python package to mirror industry-standard development practices.

## Core Thesis

This framework is built on the thesis that deep learning models can identify and exploit complex, non-linear patterns in financial time-series data that are often missed by traditional factor models. The system is designed to rigorously test this by integrating custom TensorFlow models into a robust, walk-forward validation pipeline.

## Key Architectural Features

- **Modular & Installable:** Structured as a proper Python package (`setup.py`, `requirements.txt`), ensuring clean dependencies and reusability.
- **Scalable Data Pipelines:** Features a dedicated `data_fetcher` that uses `yfinance` and saves data in the high-performance Parquet format to expedite read/write operations.
- **Advanced Feature Engineering:** Leverages Scikit-learn's `Pipeline` and `FeatureUnion` with custom transformers (`Momentum`, `Z-Score`) to create a extensible feature engineering workflow.
- **Rigorous Model Training:** The `models` package contains a training pipeline, including proper train/validation/test splitting and TensorFlow callbacks like `EarlyStopping` to prevent overfitting.
- **Walk-Forward Validation:** The framework's core is a `walk_forward.py` script that performs robust out-of-sample testing. It retrains the ML model on rolling windows of data to simulate realistic production trading environments and avoid lookahead bias.
- **Performance Analytics:** The backtesting engine uses `vectorbt` and automatically generates key performance metrics (e.g., Sharpe Ratio, Total Return) and visualizations for each validation fold.

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Spoodermon/multi_asset_backtester.git](https://github.com/Spoodermon/multi_asset_backtester.git)
    cd multi_asset_backtester
    ```

2.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```
    *(This will also install all dependencies from `requirements.txt`)*

3.  **Run the full pipeline:**
    ```bash
    python scripts/walk_forward.py
    ```

## License

MIT License