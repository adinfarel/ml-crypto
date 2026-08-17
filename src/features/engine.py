import numpy as np
import pandas as pd
from typing import List, Tuple
from src.config import FeaturesConfig

class FeatureEngine:
    '''Treat data strictly to zero-leakage guarantee.'''
    
    def __init__(self, config: FeaturesConfig):
        self.config = config
    
    def create_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        data = df.copy()
        
        # forward-log return
        # why?
        # we want predict how much return we can get instead of
        # we predict exactly number in future, because in finance
        # stock price or crypto price so volatile
        data['target_return'] = np.log(
            data['close'].shift(-self.config.return_horizon) / data['close']
        )
        
        # moving average
        feature_cols: List[str] = []
        for window in self.config.ma_windows:
            col_name = f"ma_dist_{window}"
            ma_series = data["close"].rolling(window=window).mean()
            data[col_name] = (data['close'] - ma_series) / ma_series
            feature_cols.append(col_name)
        
        # volatility
        data["log_ret_1m"] = np.log(data["close"] / data["close"].shift(1))
        vol_col = f"volatility_{self.config.volatility_window}"
        data[vol_col] = data['log_ret_1m'].rolling(window=self.config.volatility_window).std()
        feature_cols.append(vol_col)
        
        # relative strength index (rsi)
        rsi_col = f"rsi_{self.config.rsi_period}"
        data[rsi_col] = self._calculate_rsi(data["close"], period=self.config.rsi_period)
        feature_cols.append(rsi_col)
        
        # volume ratio
        data["vol_ratio_15m"] = data["volume"] / (data['volume'].rolling(15).mean() + 1e-8)
        feature_cols.append("vol_ratio_15m")
        
        # drop NaN
        data.dropna(subset=feature_cols + ["target_return"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        return data, feature_cols

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi