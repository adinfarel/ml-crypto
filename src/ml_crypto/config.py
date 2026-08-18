import os
import yaml
from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

class SystemConfig(BaseModel):
    project_name: str
    random_seed: int
    environment: str

class DataConfig(BaseModel):
    raw_path: Path
    processed_dir: Path
    reference_dir: Path
    time_column: str
    target_column: str
    train_ratio: float
    val_ratio: float
    test_ratio: float

class FeaturesConfig(BaseModel):
    schema_version: str
    ma_windows: List[int]
    rsi_period: int
    volatility_window: int
    return_horizon: int

class ModelingConfig(BaseModel):
    model_type: str
    hyperparameters: Dict[str, Any]

class DriftConfig(BaseModel):
    psi_threshold: float
    ks_alpha: float

class ArtifactsConfig(BaseModel):
    base_dir: Path
    models_dir: Path
    runs_dir: Path

class AppConfig(BaseModel):
    system: SystemConfig
    data: DataConfig
    features: FeaturesConfig
    modeling: ModelingConfig
    drift: DriftConfig
    artifacts: ArtifactsConfig
    
    @classmethod
    def load_from_yaml(cls, config_path: str = "config/config.yaml") -> "AppConfig":
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config file not found at: {config_path}")
        
        with open(config_path, 'r') as reader:
            raw_config = yaml.safe_load(reader)
        
        return cls(**raw_config)

# singleton getter
def get_config(config_path: str = 'config/config.yaml') -> AppConfig:
    return AppConfig.load_from_yaml(config_path)