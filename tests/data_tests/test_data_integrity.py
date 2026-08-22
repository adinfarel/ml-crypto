import pytest
import pandas as pd
from ml_crypto.data.splitter import TemporalDataSplitter
from ml_crypto.data.loader import BinanceDataLoader
from ml_crypto.data.storage import DataStorageManager

def test_temporal_splitter_no_overlap():
    '''Guarantees zero overlap between train, val, and test splits.'''
    dates = pd.date_range("2026-08-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame({"datetime": dates, "value": range(100)})

    splitter = TemporalDataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splitter.split(df)

    # Strict temporal checks
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

@pytest.fixture
def dummy_multi_csv_dir(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    
    df1 = pd.DataFrame({
        "open_time": [1700000000000, 1700000060000, 1700000120000],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [10.0, 15.0, 20.0],
        "close_time": [1700000059999, 1700000119999, 1700000179999],
        "quote_volume": [1000.0, 1500.0, 2000.0],
        "trades_count": [100, 150, 200],
        "taker_buy_base_vol": [5.0, 7.0, 10.0],
        "taker_buy_quote_vol": [500.0, 700.0, 1000.0],
        "ignore": [0, 0, 0]
    })
    
    df2 = pd.DataFrame({
        "open_time": [1700000120000, 1700000180000], # 1700000120000 duplicate
        "open": [102.0, 103.0],
        "high": [107.0, 108.0],
        "low": [101.0, 102.0],
        "close": [103.0, 104.0],
        "volume": [20.0, 25.0],
        "close_time": [1700000179999, 1700000239999],
        "quote_volume": [2000.0, 2500.0],
        "trades_count": [200, 250],
        "taker_buy_base_vol": [10.0, 12.0],
        "taker_buy_quote_vol": [1000.0, 1200.0],
        "ignore": [0, 0]
    })

    df1.to_csv(data_dir / "BTCUSDT-1m-2025-07.csv", index=False, header=False)
    df2.to_csv(data_dir / "BTCUSDT-1m-2025-08.csv", index=False, header=False)

    return data_dir

def test_binance_data_loader_multi_file_deduplication(dummy_multi_csv_dir):
    loader = BinanceDataLoader(dummy_multi_csv_dir)
    df = loader.load_raw_klines()

    assert len(df) == 4
    assert df["datetime"].is_monotonic_increasing
    assert "ignore" not in df.columns

def test_data_storage_manager(tmp_path, dummy_multi_csv_dir):
    loader = BinanceDataLoader(dummy_multi_csv_dir)
    df_raw = loader.load_raw_klines()

    storage_dir = tmp_path / "processed"
    storage_mgr = DataStorageManager(base_processed_dir=storage_dir)

    # Save
    saved_path = storage_mgr.save_versioned_dataset(df_raw, version="test_v1")
    assert saved_path.exists()
    assert (storage_dir / "versioned" / "test_v1" / "dataset_hash.txt").exists()

    # Load
    df_loaded = storage_mgr.load_versioned_dataset(version="test_v1")
    assert len(df_loaded) == len(df_raw)