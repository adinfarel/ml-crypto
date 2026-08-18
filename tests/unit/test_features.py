import pytest
import pandas as pd
import numpy as np
from ml_crypto.config import FeaturesConfig
from ml_crypto.features.engine import FeatureEngine

@pytest.fixture
def sample_features_config():
    return FeaturesConfig(
        schema_version="v1.0.0",
        ma_windows=[5, 15, 60],
        rsi_period=14,
        volatility_window=20,
        return_horizon=1,
    )

@pytest.fixture
def mock_kline_df():
    '''Generates 200 synthetic 1-minute bars for testing.'''
    dates = pd.date_range("2026-08-01 00:00:00", periods=200, freq="1min", tz="UTC")
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(200) * 10)
    
    df = pd.DataFrame({
        "open_time": (dates.astype("int64") // 10**6),
        "datetime": dates,
        "open": prices,
        "high": prices + 5,
        "low": prices - 5,
        "close": prices,
        "volume": np.random.uniform(10, 100, 200),
        "quote_volume": np.random.uniform(500000, 5000000, 200),
        "trades_count": np.random.randint(100, 1000, 200),
        "taker_buy_base_vol": np.random.uniform(5, 50, 200),
        "taker_buy_quote_vol": np.random.uniform(250000, 2500000, 200),
    })
    return df

def test_feature_engine_creates_expected_columns(mock_kline_df, sample_features_config):
    engine = FeatureEngine(sample_features_config)
    df_out, feature_cols = engine.create_features_and_target(mock_kline_df)

    assert "target_return" in df_out.columns
    assert len(feature_cols) == 6
    assert df_out[feature_cols].isnull().sum().sum() == 0


def test_anti_leakage_target_construction(mock_kline_df, sample_features_config):
    '''Verifies that target_return(t) strictly equals log(close(t+1) / close(t)).'''
    engine = FeatureEngine(sample_features_config)
    df_out, _ = engine.create_features_and_target(mock_kline_df)

    # Re-calculate expected return manually for a specific row
    idx = 10
    current_close = df_out.iloc[idx]["close"]
    next_close = df_out.iloc[idx + 1]["close"]
    expected_return = np.log(next_close / current_close)

    actual_return = df_out.iloc[idx]["target_return"]
    assert np.isclose(actual_return, expected_return, atol=1e-6)