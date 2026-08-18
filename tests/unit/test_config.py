import pytest
import numpy as np
from ml_crypto.config import get_config
from ml_crypto.utils import set_seed, calculate_file_hash

def test_config_loader():
    config = get_config("config/config.yaml")
    assert config.system.project_name == "ml-crypto"
    assert config.system.random_seed == 42
    assert config.data.train_ratio + config.data.val_ratio + config.data.test_ratio == 1.0

def test_deterministic_seed():
    set_seed(42)
    val_1 = [float(x) for x in list(np.random.randn(3))]
    
    set_seed(42)
    val_2 = [float(x) for x in list(np.random.randn(3))]

    assert val_1 == val_2