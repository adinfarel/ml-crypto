import os
import random
import hashlib
import numpy as np
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    '''Fix random seeds across python, numpy for reproducibility.'''
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def calculate_file_hash(file_path: Path) -> str:
    '''Calculate MD5 hash of a file for data version tracking.'''
    if not os.path.exists(file_path):
        return "FILE_NOT_FOUND"
    
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as reader:
        for chunk in iter(lambda: reader.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()