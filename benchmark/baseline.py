from narwhals import col
import numpy as np
from lightgbm import LGBMRegressor
from pathlib import Path

from benchmark.runner import BenchmarkRunner
from ml_crypto.config import get_config
from ml_crypto.data.loader import BinanceDataLoader
from ml_crypto.features.engine import FeatureEngine
from ml_crypto.data.splitter import TemporalDataSplitter
from ml_crypto.inference.batch import BatchInferenceEngine
from ml_crypto.inference.online import OnlinePredictor
from ml_crypto.models.trainer import ModelTrainer
from ml_crypto.features.schema import FEATURE_SCHEMA

def load_data_and_feat_eng():
    config = get_config('config/config.yaml')
    
    loader = BinanceDataLoader(
        config.data.raw_path
    )
    
    feature_engine = FeatureEngine(
        config.features
    )
    
    print("Preparing benchmark inputs...")
    raw_df = loader.load_raw_klines()
    
    print(
        f"Raw dataset: "
        f"{len(raw_df):,} rows x {len(raw_df.columns)} columns"
    )
    
    runner = BenchmarkRunner("load_data_and_feat_eng")
    runner.add_stage(
        "data_loading",
        loader.load_raw_klines
    )
    runner.add_stage(
        "feature_engineering",
        lambda: feature_engine.create_features_and_target(raw_df)
    )
    
    runner.run(warmup=1, iterations=5)

def training():
    config = get_config('config/config.yaml')
    
    loader = BinanceDataLoader(config.data.raw_path)
    feature_engine = FeatureEngine(config.features)
    splitter = TemporalDataSplitter(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    
    raw_df = loader.load_raw_klines()
    feature_df, feature_cols = feature_engine.create_features_and_target(raw_df)
    train_df, val_df, test_df = splitter.split(feature_df)
    
    X_train = train_df[feature_cols]
    y_train = train_df[config.data.target_column]
    
    def create_model():
        return LGBMRegressor(
            **config.modeling.hyperparameters,
        )
    
    def train():
        model = create_model()
        model.fit(X_train, y_train)
        return model
    
    runner = BenchmarkRunner(name='training')
    runner.add_stage(
        "model_fit",
        train
    )
    
    runner.run(
        warmup=1,
        iterations=5
    )

def model_trainer():
    config = get_config('config/config.yaml')
    
    loader = BinanceDataLoader(config.data.raw_path)
    feature_engine = FeatureEngine(config.features)
    splitter = TemporalDataSplitter(
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    
    raw_df = loader.load_raw_klines()
    feature_df, feature_cols = feature_engine.create_features_and_target(raw_df)
    train_df, val_df, test_df = splitter.split(feature_df)
    
    trainer = ModelTrainer(config)
    
    def train_pipeline():
        return trainer.train_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
        )
    
    runner = BenchmarkRunner(name="model_trainer")
    runner.add_stage(
        "train_and_evaluate",
        train_pipeline
    )
    
    runner.run(
        warmup=1,
        iterations=5
    )

_PRODUCTION_MODEL_PATH: Path = Path(
    get_config().artifacts.models_dir / "production_model.pkl"
)

def batch_inference():
    config = get_config('config/config.yaml')
    
    loader = BinanceDataLoader(config.data.raw_path)
    feature_engine = FeatureEngine(config.features)
    
    raw_df = loader.load_raw_klines()
    
    feature_df, feature_cols = (
        feature_engine.create_features_and_target(raw_df)
    )
    
    inference_df = feature_df[feature_cols].copy()
    
    print(
        f"Inference dataset: "
        f"{len(inference_df):,} rows x "
        f"{len(inference_df.columns)} features"
    )
    
    engine = BatchInferenceEngine(_PRODUCTION_MODEL_PATH)
    
    def model_predict():
        X = inference_df[feature_cols]
        return engine.model.predict(X)

    def batch_predict():
        return engine.predict(feature_df)
    
    # breakdown
    def validate():
        missing_features = [
            col
            for col in FEATURE_SCHEMA.required_features
            if col not in inference_df.columns
        ]
        
        if missing_features:
            raise ValueError(
                f"missing required features: {missing_features}"
            )
    
    def df_copy():
        return inference_df.copy()

    def feature_selection():
        return inference_df[FEATURE_SCHEMA.required_features]
    
    def output_construction():
        output_df = inference_df.copy()

        X = inference_df[
            FEATURE_SCHEMA.required_features
        ]

        preds = engine.model.predict(X)

        output_df["predicted_return"] = preds
        output_df["signal"] = (
            output_df["predicted_return"] > 0
        ).astype(int)

        return output_df
    
    runner = BenchmarkRunner(name='batch_inference')
    runner.add_stage(
        'model_predict',
        model_predict,
    )
    runner.add_stage(
        'batch_inference',
        batch_predict
    )
    
    runner.add_stage(
    "validation",
    validate,
    )

    runner.add_stage(
        "dataframe_copy",
        df_copy,
    )

    runner.add_stage(
        "feature_selection",
        feature_selection,
    )

    runner.add_stage(
        "output_construction",
        output_construction,
    )
    
    runner.run(
        warmup=1, iterations=10
    )

def online_inference():
    predictor = OnlinePredictor(_PRODUCTION_MODEL_PATH)
    runner = BenchmarkRunner(name='online_inference')
    
    feature_dict = {    
        "ma_dist_5": 0.001,
        "ma_dist_15": -0.002,
        "ma_dist_30": 0.003,
        "volatility_30": 0.0015,
        "rsi_14": 52.3,
        "vol_ratio_15m": 1.12,
    }


    def online_predict():
        return predictor.predict_sample(feature_dict)


    runner.add_stage(
        "online_predict",
        online_predict,
    )
    
    runner.run(
        warmup=1, iterations=5
    )

if __name__ == "__main__":
    online_inference()