import pytest
import pandas as pd
import numpy as np
from ml_crypto.config import get_config
from ml_crypto.models.trainer import ModelTrainer

@pytest.fixture
def mock_dataset():
    '''Generates synthetic dataset for model training test.'''
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "ma_dist_5": np.random.randn(n),
        "ma_dist_15": np.random.randn(n),
        "ma_dist_60": np.random.randn(n),
        "volatility_20": np.abs(np.random.randn(n)),
        "rsi_14": np.random.uniform(20, 80, n),
        "vol_ratio_15m": np.random.uniform(0.5, 2.0, n),
        "target_return": np.random.randn(n) * 0.01,
    })
    return df

def test_model_trainer_execution(mock_dataset, tmp_path):
    config = get_config("config/config.yaml")
    
    config.artifacts.models_dir = tmp_path / "models"
    config.artifacts.runs_dir = tmp_path / "runs"
    
    trainer = ModelTrainer(config)
    feature_cols = ["ma_dist_5", "ma_dist_15", "ma_dist_60", "volatility_20", "rsi_14", "vol_ratio_15m"]

    train_df = mock_dataset.iloc[:200]
    val_df = mock_dataset.iloc[200:]

    model, manifest = trainer.train_and_evaluate(train_df, val_df, feature_cols)

    assert "val_rmse" in manifest["metrics"]
    assert "directional_accuracy" in manifest["metrics"]
    assert manifest["metrics"]["directional_accuracy"] >= 0.0
    assert (tmp_path / "models" / "production_model.pkl").exists()