import numpy as np
import pandas as pd
from typing import List, Tuple
from ml_crypto.config import FeaturesConfig

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
            col_dist = f"ma_dist_{window}"
            col_zscore = f"ma_zscore_{window}"
            
            ma_series = data["close"].rolling(window=window).mean()
            std_series = data["close"].rolling(window=window).std()
            
            data[col_dist] = (data['close'] - ma_series) / (ma_series + 1e-8)
            data[col_zscore] = (data['close'] - ma_series) / (std_series + 1e-8)
            feature_cols.extend([col_dist, col_zscore])
        
        # volatility
        data["log_ret_1m"] = np.log(data["close"] / data["close"].shift(1))
        vol_col = f"volatility_{self.config.volatility_window}"
        data[vol_col] = data['log_ret_1m'].rolling(window=self.config.volatility_window).std()
        feature_cols.append(vol_col)
        
        # parkinson
        data["vol_parkinson_15"] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * (np.log(data["high"] / (data["low"] + 1e-8)) ** 2)
        ).rolling(window=15).mean()
        feature_cols.append("vol_parkinson_15")
        
        # relative strength index (rsi)
        rsi_col = f"rsi_{self.config.rsi_period}"
        data[rsi_col] = self._calculate_rsi(data["close"], period=self.config.rsi_period)
        feature_cols.append(rsi_col)
        
        # multi-horizon log-return
        for lag in [3, 5, 15]:
            ret_col = f"log_ret_{lag}m"
            data[ret_col] = np.log(data['close'] / df['close'].shift(lag))
            feature_cols.append(ret_col)
        
        # volume ratio
        data["vol_ratio_15m"] = data["volume"] / (data['volume'].rolling(15).mean() + 1e-8)
        feature_cols.append("vol_ratio_15m")
        
        if "taker_buy_base_vol" in data.columns:
            data['taker_buy_ratio'] = data['taker_buy_base_vol'] / (data['volume'] + 1e-8)
            feature_cols.append('taker_buy_ratio')
        
        if 'trades_count' in data.columns:
            data['avg_trade_size'] = data['volume'] / (data['trades_count'] + 1e-8)
            data['trade_size_ratio'] = data['avg_trade_size'] / (data['avg_trade_size'].rolling(window=15).mean())
            feature_cols.append('trade_size_ratio')
        
        # drop NaN
        data.dropna(subset=feature_cols + ["target_return"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        return data, feature_cols

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = (delta.where(delta > 0, 0))
        loss  = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi