import pandas as pd
from pathlib import Path

from ml_crypto.utils import calculate_file_hash

class DataStorageManager:
    '''Managed data into parquet format (columnar)'''
    
    def __init__(self, base_processed_dir: Path = Path('data/processed')):
        self.base_dir = base_processed_dir
    
    def save_versioned_dataset(self, df: pd.DataFrame, version: str = "v1_12months") -> Path:
        version_dir = self.base_dir / "versioned" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = version_dir / "merged_klines.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        
        dataset_hash = calculate_file_hash(output_file)
        with open(version_dir / "dataset_hash.txt", "w") as f:
            f.write(dataset_hash)
        
        return output_file
    
    def load_versioned_dataset(self, version: str = "v1_12months") -> pd.DataFrame:
        file_path = self.base_dir / "versioned" / version / "merged_klines.parquet"
        if not file_path.exists():
            raise ValueError(f"versioned dataset missing at {file_path}")
        return pd.read_parquet(file_path)