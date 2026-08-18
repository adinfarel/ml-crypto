from ml_crypto.config import get_config
from ml_crypto.utils import set_seed
from ml_crypto.data.loader import BinanceDataLoader
from ml_crypto.data.splitter import TemporalDataSplitter
from ml_crypto.features.engine import FeatureEngine
from ml_crypto.models.trainer import ModelTrainer

def main():
    config = get_config("config/config.yaml")
    set_seed(config.system.random_seed)
    
    print(f"=== PROJECTS: {config.system.project_name} ===")
    
    # ingestion
    loader  = BinanceDataLoader(config.data.raw_path)
    raw_df  = loader.load_raw_klines()
    
    # features
    features_engine = FeatureEngine(config.features)
    processed_df, feature_cols = features_engine.create_features_and_target(raw_df)
    
    # temporal split
    splitter = TemporalDataSplitter(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    train_df, val_df, test_df = splitter.split(processed_df)
    
    # training
    trainer = ModelTrainer(config)
    print("\nTraining model and evaluating...")
    _, manifest = trainer.train_and_evaluate(train_df, val_df, feature_cols)
    
    print(f"\n=== SUMMARY: {config.system.project_name} ===")
    print(f"Run ID              : {manifest['run_id']}")
    print(f"Training RMSE       : {manifest['metrics']['train_rmse']:.6f}")
    print(f"Validation RMSE     : {manifest['metrics']['val_rmse']:.6f}")
    print(f"Directional Acc     : {manifest['metrics']['directional_accuracy'] * 100:.2f}%")
    print(f"Promoted to Prod    : {manifest['is_promoted_to_production']}")

if __name__ == "__main__":
    main()