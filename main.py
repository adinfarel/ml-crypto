from src.config import get_config
from src.utils import set_seed
from src.data.loader import BinanceDataLoader
from src.data.splitter import TemporalDataSplitter
from src.features.engine import FeatureEngine


def main():
    config = get_config("config/config.yaml")
    set_seed(config.system.random_seed)

    print(f"=== Project: {config.system.project_name} ===")
    
    # 1. Ingestion
    loader = BinanceDataLoader(config.data.raw_path)
    raw_df = loader.load_raw_klines()
    print(f"[A2] Raw Data Loaded: {len(raw_df):,} rows")

    # 2. Feature Engineering
    feature_engine = FeatureEngine(config.features)
    processed_df, feature_cols = feature_engine.create_features_and_target(raw_df)
    print(f"[A3] Feature Engineering Complete: {len(feature_cols)} features created")
    print(f"     Features: {feature_cols}")
    print(f"     Processed Dataset Shape: {processed_df.shape}")

    # 3. Temporal Split
    splitter = TemporalDataSplitter(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    train_df, val_df, test_df = splitter.split(processed_df)

    print("\n[A2/A3] Strict Temporal Split Summary:")
    print(f"     Train Set : {len(train_df):,} rows | {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
    print(f"     Val Set   : {len(val_df):,} rows  | {val_df['datetime'].min()} -> {val_df['datetime'].max()}")
    print(f"     Test Set  : {len(test_df):,} rows  | {test_df['datetime'].min()} -> {test_df['datetime'].max()}")


if __name__ == "__main__":
    main()