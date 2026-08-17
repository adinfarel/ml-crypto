from typing import Tuple
import pandas as pd

class TemporalDataSplitter:
    '''Chronological time-series splitting w/out shuffling.'''
    
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
        if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
            raise ValueError("train, test, and val ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''Split DataFrame sequentially based on timestamp ordering.'''
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.val_ratio + self.train_ratio))
        
        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)
        
        self._verify_no_overlap(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    @staticmethod
    def _verify_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        '''Sanity check to guarantee zero temporal overlap.'''
        if not train_df.empty and not val_df.empty:
            assert train_df["datetime"].max() < val_df["datetime"].min(), "train-val temporal overlap detected."
            
        if not test_df.empty and not val_df.empty:
            assert val_df["datetime"].max() < test_df["datetime"].min(), "val-test temporal overlap detected."