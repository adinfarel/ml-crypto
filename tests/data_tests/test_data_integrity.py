import pytest
import pandas as pd
from ml_crypto.data.splitter import TemporalDataSplitter

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