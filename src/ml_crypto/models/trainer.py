import time
import uuid
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from ml_crypto.config import AppConfig
from ml_crypto.utils import calculate_file_hash
from ml_crypto.features.schema import FEATURE_SCHEMA
from ml_crypto.models.registry import ModelRegistry

class ModelTrainer:
    '''Training pipeline.'''
    
    __slots__ = ("config", "registry",)
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.registry = ModelRegistry(
            models_dir=self.config.artifacts.models_dir,
            runs_dir=self.config.artifacts.runs_dir
        )
    
    def train_and_evaluate(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list
    ) -> Tuple[Any, Dict[str, Any]]:
        '''Train LightGBM model'''
        X_train = train_df[feature_cols]
        y_train = train_df[self.config.data.target_column]
        
        X_val = val_df[feature_cols]
        y_val = val_df[self.config.data.target_column]
        # setup model
        model = LGBMRegressor(**self.config.modeling.hyperparameters)
        # fit (training) model
        model.fit(X_train, y_train)
        # predict
        preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        # evaluate
        # how large difference predict and actual
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        # how often model correct predict sign (+/-)
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val)))
        
        metrics = {
            "train_rmse": tr_rmse,
            "val_rmse": rmse,
            "directional_accuracy": dir_acc,
        }
        
        # manifest
        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        manifest = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "environment": self.config.system.environment,
            "seed": self.config.system.random_seed,
            "dataset_hash": calculate_file_hash(self.config.data.raw_path),
            "feature_schema_version": FEATURE_SCHEMA.version,
            "features_used": feature_cols,
            "model_type": self.config.modeling.model_type,
            "hyperparameters": self.config.modeling.hyperparameters,
            "metrics": metrics,
        }
        
        # artifacts
        self.registry.save_run(run_id, model, manifest)
        is_promoted = self.registry.promote_if_better(run_id, current_metric=dir_acc)
        manifest["is_promoted_to_production"] = is_promoted
        
        return model, manifest