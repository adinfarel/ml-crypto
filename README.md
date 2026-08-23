# Quant ML Pipeline (`quant-ml-pipeline`)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![DVC](https://img.shields.io/badge/Data_Control-DVC-945DD6.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end quantitative ML pipeline for high-frequency crypto time-series data (Binance BTC/USDT 1-minute klines). Built around strict **zero-lookahead / anti-leakage** feature engineering, statistical drift detection, a unified pipeline orchestrator, model registry with promotion gating, and FastAPI + Streamlit serving.

> **Status: active development.** Core pipeline (ingestion -> features -> temporal split -> drift check -> train -> registry) runs end-to-end and is covered by unit/integration/data/model tests. Docker/CI packaging is still being hardened.

---

## System Architecture

```text
                    +----------------------------+
                    |   Binance Raw Kline Data   |
                    +--------------+-------------+
                                   |
                                   v
                    +----------------------------+
                    |  Data Loader & Validation  |
                    | (Timestamp / Schema Check) |
                    +--------------+-------------+
                                   |
                                   v
                    +----------------------------+
                    | Leakage-Safe Feature Eng.  |
                    |  (MA Dist, RSI, Vol, VolR) |
                    +--------------+-------------+
                                   |
                                   v
                    +----------------------------+
                    | Strict Temporal Splitter   |
                    |   (Train / Val / Test)     |
                    +--------------+-------------+
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
+------------------------------+          +------------------------------+
| Statistical Drift Detector   |          |  Unified Pipeline Runner     |
|     (PSI & KS-Test)          |          |  (LightGBM + Early Stopping) |
+------------------------------+          +--------------+---------------+
              |                                         |
              +--------------------+--------------------+
                                   |
                                   v
                    +------------------------------+
                    | Model Registry &             |
                    | Promotion Evaluation Gate    |
                    +--------------+---------------+
                                   |
                                   v
                    +------------------------------+
                    |   FastAPI Serving Endpoint   |
                    +--------------+---------------+
                                   |
                                   v
                    +------------------------------+
                    |         UI Dashboard         |
                    +------------------------------+
```

---

## Project Structure

```text
quant-ml-pipeline/
├── .dvc/                      # Data Version Control internal state
│   └── config
├── .dvcignore
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # CI/CD automated testing pipeline
├── config/
│   └── config.yaml            # Single source of truth configuration
├── data/
│   ├── raw/                   # Raw CSV/Parquet kline data (tracked by DVC)
│   ├── raw.dvc                # DVC pointer file for raw data
│   ├── processed/             # Processed feature sets
├── artifacts/
│   ├── models/                # Model registry (.pkl binaries tracked by DVC)
│   └── runs/                  # JSON run manifests (see below)
├── src/
│   ├── __init__.py
│   ├── config.py              # Typed Pydantic config loader
│   ├── utils.py               # Seed control, recursive MD5 hashing
│   ├── data/
│   │   ├── loader.py          # Data ingestion engine
│   │   ├── schema.py          # Schema validation
│   │   ├── splitter.py        # Strict chronological temporal splitter
│   │   └── drift_detector.py  # PSI & Kolmogorov-Smirnov drift checks
│   ├── features/
│   │   ├── engine.py          # Vectorized feature engine
│   │   └── schema.py          # Feature schema versioning
│   ├── models/
│   │   ├── trainer.py         # LightGBM training with early stopping
│   │   └── registry.py        # Model registry & atomic promotion gate
│   ├── pipeline/
│   │   └── runner.py          # Unified pipeline runner & orchestrator
│   └── inference/
│       ├── batch.py           # Offline batch predictor
│       └── online.py          # Low-latency single-sample predictor
├── serving/
│   ├── app.py                 # FastAPI serving endpoint
│   └── schemas.py             # Pydantic API validation schemas
├── ui/
│   └── index.html 
|   └── style.css
|   └── script.js              # Three vanilla html, css, and js
├── tests/
│   ├── unit/                  # Feature math & config unit tests
│   ├── integration/           # Serving API integration tests
│   ├── data_tests/            # Data integrity & temporal split tests
│   └── model_tests/           # Model sanity & determinism tests
├── .gitignore
├── Dockerfile                 # Multi-stage containerization (WIP, see below)
├── docker-compose.yml         # Multi-container service setup (WIP, see below)
├── Makefile                   # Standardized CLI shortcuts
├── pyproject.toml             # Project metadata & dependency spec
├── main.py                    # Main CLI entrypoint
└── README.md
```

---

## Key Features

1. **Zero-Lookahead Bias Guarantee** — chronological splitting (`T_train < T_val < T_test`) and `shift(1)` indicator calculations, so no feature ever sees future information.
2. **Unified Pipeline Runner** — one orchestrator for standard training (`--mode train`) and drift-triggered automated retraining (`--mode drift-retrain`).
3. **Early Stopping** — LightGBM training monitors validation loss to cut off overfitting early.
4. **Statistical Drift Detection** — feature distribution shifts monitored via **Population Stability Index (PSI)** and **Kolmogorov–Smirnov test**, used to trigger retraining on regime shifts.
5. **Atomic Model Registry** — candidate models are evaluated against the current production model and promoted via atomic file-copy (`shutil.copyfile`), so a promotion can never leave the registry in a half-written state.
6. **Three Vanilla UI** — visual interface for model performance, drift analytics, and signal inspection.

---

## Quickstart

### 1. Environment setup

```bash
git clone https://github.com/adinfarel/quant-ml-pipeline.git
cd quant-ml-pipeline

python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installs the package in editable mode with dev dependencies
# (pytest, linting, etc.) as declared in pyproject.toml
pip install -e ".[dev]"
```

### 2. Running the pipeline

```bash
# Standard end-to-end training
python main.py --mode train

# Drift-triggered automated retraining
python main.py --mode drift-retrain

# Custom config / dataset override
python main.py -c config/config.yaml --mode train --data-path data/raw/BTCUSDT-1m.csv
```

### 3. Or via the Makefile

```bash
make install     # pip install -e ".[dev]"
make test        # pytest -v
make train       # run standard training
make run-api     # uvicorn serving.app:app --reload --port 8000
make docker-build
make docker-run  # docker-compose up --build -d
make docker-stop
make clean       # remove __pycache__ / *.pyc
```

---

## Testing

```bash
pytest -v
# or
make test
```

Test suite is split into `unit/` (feature math, config), `integration/` (serving API), `data_tests/` (schema + temporal split integrity), and `model_tests/` (sanity + determinism).

---

## Serving

```bash
# FastAPI endpoint
uvicorn serving.app:app --reload --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
```

---

## Example Run Manifest

Every run writes a JSON manifest to `artifacts/runs/` capturing the exact config, feature set, and metrics for reproducibility:

```json
{
  "run_id": "1787411311_ec53d9",
  "timestamp": "2026-08-22 15:08:31 UTC",
  "environment": "development",
  "seed": 42,
  "dataset_hash": "a20b6f1d138d031b026f4c3cd28a2040",
  "feature_schema_version": "v1.0.0",
  "features_used": [
    "ma_dist_5", "ma_zscore_5",
    "ma_dist_15", "ma_zscore_15",
    "ma_dist_60", "ma_zscore_60",
    "volatility_20", "vol_parkinson_15",
    "rsi_14",
    "log_ret_3m", "log_ret_5m", "log_ret_15m",
    "vol_ratio_15m", "taker_buy_ratio", "trade_size_ratio"
  ],
  "model_type": "lightgbm",
  "hyperparameters": {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 42
  },
  "metrics": {
    "train_rmse": 0.0006316639379092431,
    "val_rmse": 0.000480696634985405,
    "directional_accuracy": 0.49679456164118446,
    "information_ratio": 9.417060464272403,
    "best_iteration": 8
  }
}
```

---

## 🛠 Project Status & Known Issues

This project is under active development. Current focus areas:

- **Docker packaging** — `Dockerfile` / `docker-compose.yml` build but are not yet fully validated end-to-end; expect rough edges until this is hardened.
- **CI workflow** — `ci_cd.yml` runs the test suite on push; deployment stages are not wired up yet.
- **Model performance** — current `directional_accuracy` is close to 0.5 (chance level) on the validation window above; this is an early checkpoint, not a claim of predictive edge. Treat metrics in `artifacts/runs/` as engineering benchmarks (pipeline correctness, reproducibility, latency) rather than trading-readiness signals.

Contributions/fixes to any of the above are being made incrementally — see commit history for latest changes.

---