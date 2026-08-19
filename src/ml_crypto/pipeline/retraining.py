import logging
from typing import Dict, Any
from ml_crypto.config import get_config
from ml_crypto.data import drift_detector
from ml_crypto.data.loader import BinanceDataLoader
from ml_crypto.features.engine import FeatureEngine
from ml_crypto.data.splitter import TemporalDataSplitter
from ml_crypto.data.drift_detector import DriftDetector
from ml_crypto.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

class RetrainingController:
    '''Auto retraining when drift detected.'''
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = get_config(config_path)
    
    def run_pipeline(self, force_retrain: bool = False) -> Dict[str, Any]:
        loader = BinanceDataLoader(self.config.data.raw_path)
        raw_df = loader.load_raw_klines()
        
        feature_engine = FeatureEngine(self.config.features)
        processed_df, feature_cols = feature_engine.create_features_and_target(raw_df)
        
        splitter = TemporalDataSplitter(
            self.config.data.train_ratio,
            self.config.data.val_ratio,
            self.config.data.test_ratio,
        )
        train_df, val_df, test_df = splitter.split(processed_df)
        
        _drift_detector = DriftDetector()
        drift_report = _drift_detector.detect_drift(train_df, test_df, feature_cols)
        
        should_retrain = force_retrain or drift_report["has_drift"]
        
        if should_retrain:
            logger.info("Retraining triggered (force=%s, drift=%s)", force_retrain, drift_report['has_drift'])
            trainer = ModelTrainer(self.config)
            _, manifest = trainer.train_and_evaluate(train_df, val_df, feature_cols)
            return {
                "status":"retrained",
                "manifest": manifest,
                "drift_report": drift_report
            }
        
        logger.info("No drift detected; retraining skipped.")
        return {"status": "skipped", "reason": "No feature drift detected", "drift_report": drift_report}