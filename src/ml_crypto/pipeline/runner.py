import logging
import pandas as pd
from typing import Dict, Any, Optional

from ml_crypto.config import AppConfig, get_config
from ml_crypto.data.loader import BinanceDataLoader
from ml_crypto.features.engine import FeatureEngine
from ml_crypto.data.splitter import TemporalDataSplitter
from ml_crypto.data.drift_detector import DriftDetector
from ml_crypto.models.trainer import ModelTrainer

logger = logging.getLogger("ml_crypto.pipeline.runner")

class PipelineRunner:
    
    def __init__(self, config: Optional[AppConfig] = None, config_path: str = 'config/config.yaml'):
        self.config = config or get_config(config_path)
    
    def run(
        self,
        mode: str = 'train',
        data_path_override: Optional[str] = None
    ) -> Dict[str, Any]:
        data_path = data_path_override or self.config.data.raw_path
        logger.info(f"executing pipeline [mode: {mode.upper()}] using data: {data_path}")
        
        loader = BinanceDataLoader(data_path)
        raw_df = loader.load_raw_klines()
        
        feature_engine = FeatureEngine(self.config.features)
        processed_df, feature_cols = feature_engine.create_features_and_target(raw_df)

        splitter = TemporalDataSplitter(
            self.config.data.train_ratio,
            self.config.data.val_ratio,
            self.config.data.test_ratio,
        )
        train_df, val_df, test_df = splitter.split(processed_df)

        drift_report: Dict[str, Any] = {"has_drift": False, "details": {}}
        should_train = True
        
        if mode == 'drift-retrain':
            detector = DriftDetector()
            drift_report = detector.detect_drift(
                reference_df=train_df,
                current_df=test_df,
                feature_cols=feature_cols
            )
            should_train = drift_report.get("has_drift", False)
            
            if not should_train:
                logger.info("no feature drift detected. training skipped as requested by 'drift-retrain' mode.")
                return {
                    "status": "skipped",
                    "reason": "No feature drift detected",
                    "drift_report": drift_report
                }
                
        logger.info(f"triggering model training (reason: mode={mode.upper()}, drift={drift_report.get('has_drift')})...")
        trainer = ModelTrainer(self.config)
        model, manifest = trainer.train_and_evaluate(train_df, val_df, feature_cols)
        
        X_test = test_df[feature_cols]
        y_test = test_df[self.config.data.target_column]
        test_preds = model.predict(X_test)
        
        non_zero_mask = y_test != 0
        test_dir_acc = float((((test_preds[non_zero_mask] > 0) == (y_test[non_zero_mask] > 0)).mean())) if non_zero_mask.sum() > 0 else 0.5
        manifest["metrics"]["test_directional_accuracy"] = test_dir_acc

        return {
            "status": "success",
            "mode": mode,
            "manifest": manifest,
            "drift_report": drift_report
        }