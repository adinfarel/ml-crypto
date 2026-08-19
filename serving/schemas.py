from pydantic import BaseModel, Field

class FeatureVectorInput(BaseModel):
    ma_dist_5: float = Field(..., example=0.002)
    ma_dist_15: float = Field(..., example=0.005)
    ma_dist_60: float = Field(..., example=-0.01)
    volatility_20: float = Field(..., example=0.0015)
    rsi_14: float = Field(..., example=55.4)
    vol_ratio_15m: float = Field(..., example=1.2)


class PredictionResponse(BaseModel):
    predicted_return: float
    signal: str
    status: str = "success"