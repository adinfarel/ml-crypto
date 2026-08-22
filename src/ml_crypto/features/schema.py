from typing import List
from pydantic import BaseModel

class FeatureSchema(BaseModel):
    version: str = "v1.0.0"
    required_features: List[str] = [
        "ma_dist_5",
        "ma_zscore_5",
        "ma_dist_15",
        "ma_zscore_15",
        "ma_dist_60",
        "ma_zscore_60",
        "volatility_20",
        "vol_parkinson_15",
        "rsi_14",
        "log_ret_3m",
        "log_ret_5m",
        "log_ret_15m",
        "vol_ratio_15m",
        "taker_buy_ratio",
        "trade_size_ratio",
    ]

FEATURE_SCHEMA = FeatureSchema()