import pandas as pd
import numpy as np
from pathlib import Path
from ml_crypto.data.schema import SCHEMA

class BinanceDataLoader:
    
    def __init__(self, raw_filepath: Path):
        self.filepath = raw_filepath
        if not self.filepath.exists():
            raise FileNotFoundError(f"raw data file not found at: {self.filepath}")
    
    def load_raw_klines(self) -> pd.DataFrame:
        '''Loads CSV w/out headers, assigns schema, and enforces correct datatypes.'''
        df = pd.read_csv(
            self.filepath,
            header=None,
            names=SCHEMA.COLUMNS,
            dtype={col: "float64" for col in SCHEMA.NUMERIC_COLUMNS}
        )
        
        # drop unused column
        df.drop(columns=["ignore"], inplace=True, errors="ignore")
        # parse timestamp (detect microseconds and miliseconds automatically)
        df["datetime"] = self._parse_timestamp(df["open_time"])
        # sort strictly by time to maintain temporal integrity
        df.sort_values(by="datetime", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        self._validate_integrity(df)
        return df
    
    @staticmethod
    def _parse_timestamp(ts_series: pd.Series) -> pd.Series:
        '''Handles both miliseconds (13 digits) and microseconds (16 digits)'''
        sample_val = ts_series.loc[0]
        # microseconds check
        if sample_val > 1e14:
            unit = "us"
        
        else:
            unit = "ms"
        
        return pd.to_datetime(ts_series, unit=unit, utc=True)

    @staticmethod
    def _validate_integrity(df: pd.DataFrame) -> None:
        '''Data sanity and integrity check.'''
        if df.empty:
            raise ValueError(f"loaded DataFrame is empty.")
        
        # check duplicate timestamps
        duplicates = df["datetime"].duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"data integrity failed: found {duplicates} duplicate timestamps.")
        
        # check missing values
        null_count = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
        if null_count > 0:
            raise ValueError(f"data integrity failed: found {null_count} missing values in core columns.")