# CoinGuard: Open-source Financial ML for Digital Asset Risk

English | [简体中文](README.zh-CN.md)

CoinGuard is an open-source financial machine learning project for digital asset risk analytics. It detects high-risk scenarios in crypto markets using engineered market microstructure and technical indicators, trained with XGBoost on temporally split data. The project aims for clarity, reproducibility, and practical deployment as a research baseline or production component.

## Key Features

- Robust data pipeline with symbol discovery and OHLCV aggregation
- Rich feature engineering: returns, volatility, rolling stats, TA indicators (RSI, MACD, Bollinger Bands, ATR)
- Time-aware target creation and temporal splits to prevent leakage
- Class imbalance handling via `scale_pos_weight`
- Hyperparameter search on validation ROC AUC
- Clear evaluation with confusion matrix, classification report, and AUC-ROC

## Repository Structure

- `download.py`: Fetches symbols, 1h klines, and long/short ratios, merges into `data/crypto_klines_data.csv`.
- `feature_engineering.py`: Builds features and saves `data/features_crypto_data.csv`.
- `train_model.py`: Creates target, splits temporally, tunes and trains XGBoost, evaluates, and plots top-20 feature importance.
- `data/`: CSV artifacts.
- `reports/`: Experiment logs and notes.

## Quickstart

1) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+

2) Download market data

```bash
python download.py
```

3) Build features

```bash
python feature_engineering.py
```

4) Train and evaluate

```bash
python train_model.py
```

Outputs include console metrics and a feature-importance plot.

## Compute Budget

You can trade off speed, cost, and accuracy via these knobs:

- Hyperparameter search budget: `CONFIG["tuning"]["max_combinations"]` in `train_model.py` (lower = faster/cheaper)
- Model size/training time: reduce `n_estimators`, `max_depth`, or increase `learning_rate`
- Evaluation overhead: keep `thresholds_to_test` short
- Data volume: limit downloaded symbols or rows before feature engineering (smaller datasets train faster)

For quick iterations, start with a small `max_combinations` (e.g., 10–20) and moderate `n_estimators` (e.g., 800), then scale up.

## Entry Conditions (optional, for signal deployment)

If you use CoinGuard to drive trading signals, you may enforce additional entry filters:

- Fee rate must be less than or equal to +0.0100% (non-positive to +0.0100%)
- Current day gain must be greater than the rolling 24h gain

Notes

- Define consistent units for fee rate (e.g., percent vs fraction) and returns.
- Ensure the necessary inputs are available in your data feed before enforcing filters.

## Data Sources

- Tickers: `https://api.lewiszhang.top/ticker/24hr`
- Klines: `https://api.lewiszhang.top/klines?symbol=BTCUSDT&interval=1h&limit=1000`
- Long/Short Account Ratio: `https://api.lewiszhang.top/topLongShortAccountRatio`
- Long/Short Position Ratio: `https://api.lewiszhang.top/topLongShortPositionRatio`

## Model Configuration (default)

- Algorithm: XGBoost binary classifier
- Lookahead horizon: 1 hour
- Risk definition: next-hour drop worse than −10%
- Validation metric: ROC AUC
- Threshold for classification: 0.6 (configurable)

See `CONFIG` in `train_model.py` to adjust parameters.

## Evaluation Results (example)

From a representative run (threshold = 0.6):

- Accuracy: 0.9935
- AUC-ROC: 0.9545
- Confusion Matrix (rows = true, cols = pred):

```
[[56386   282]
 [   89    60]]
```

- Class Report (precision / recall / f1 / support)
  - Class 0 (Low risk): 0.9984 / 0.9950 / 0.9967 / 56668
  - Class 1 (High risk): 0.1754 / 0.4027 / 0.2444 / 149

Notes

- This is a highly imbalanced classification problem; recall/precision trade-offs are controlled by the decision threshold.
- Lower thresholds generally increase recall (catch more risks) at the cost of precision (more false alarms).

## Top-20 Feature Importance (example)

The training script plots and prints the top-20 features by importance. Typical high-impact features include:

- `BBB_20_2.0` (Bollinger Band width)
- `number_of_trades`
- `RSI`
- `BBP_20_2.0` (Bollinger Band position)
- `BBL_20_2.0` / `BBM_20_2.0` / `BBU_20_2.0`
- `return_1h`, `log_return_1h`, `lag_return_*h`
- Rolling means and volatilities of `close`

Run `train_model.py` to regenerate the plot for your dataset and configuration.

## Reproducibility & Compliance

- Temporal splits and target creation are symbol-aware and avoid lookahead leakage.
- Random seed is fixed in model defaults for repeatability.
- This repository is for research and educational purposes only and does not constitute financial advice.
- Ensure compliance with your jurisdiction’s regulations and data provider terms of service.

## Security & Safety

- APIs are public endpoints; avoid committing secrets.
- Review and sandbox any downstream trading logic—this project is purely analytical.
- Validate data integrity before training; unexpected schema changes may break pipelines.

## Contributing

Contributions are welcome! Please:

- Open an issue to discuss substantial changes.
- Follow clear commit messages and include before/after metrics when relevant.
- Add tests or runnable examples where appropriate.

## License

This project is released under the terms of the license in `LICENSE`.
