import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

from ml_crypto.features.schema import FEATURE_SCHEMA

class BatchInferenceEngine:
    '''Batch prediction, time to use corressponding scheduling'''
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"production model not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_features = [
            col for col in FEATURE_SCHEMA.required_features if col not in df.columns
        ]
        
        if missing_features:
            raise ValueError(f"missing required features for inference: {missing_features}")
        
        output_df = df.copy()
        X = df[FEATURE_SCHEMA.required_features]
        output_df["predicted_return"] = self.model.predict(X)
        output_df["signal"] = (output_df["predicted_return"] > 0).astype(int)  # 1 = buy, 0 = sell
        return output_df