import joblib
from pathlib import Path
from typing import Dict, Any


class OnlinePredictor:
    '''Online prediction, use when model call'''

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"production model binary missing at {self.model_path}")
        self.model = joblib.load(self.model_path)
        
    def predict_sample(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        import pandas as pd
        X = pd.DataFrame([feature_dict])
        
        pred_return = float(self.model.predict(X)[0])
        signal = "BUY" if pred_return > 0 else "SELL"

        return {
            "predicted_return": round(pred_return, 6),
            "signal": signal,
        }