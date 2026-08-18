from typing import List
from pydantic import BaseModel

class FeatureSchema(BaseModel):
    version: str = "v1.0.0"
    required_features: List[str] = [
        "ma_dist_5",
        "ma_dist_15",
        "ma_dist_60",
        "volatility_20",
        "rsi_14",
        "vol_ratio_15m"
    ]

FEATURE_SCHEMA = FeatureSchema()