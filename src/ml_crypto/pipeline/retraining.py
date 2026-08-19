import logging
from typing import Dict, Any
from ml_crypto.config import get_config
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