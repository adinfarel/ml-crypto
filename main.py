from src.config import get_config
from src.utils import set_seed, calculate_file_hash
from src.data.loader import BinanceDataLoader


def main():
    config = get_config("config/config.yaml")
    set_seed(config.system.random_seed)

    print(f"=== Project: {config.system.project_name} ===")
    print(f"Loading data from: {config.data.raw_path}")

    # Hash check
    data_hash = calculate_file_hash(config.data.raw_path)
    print(f"Raw Data MD5 Hash: {data_hash}")

    # Load & Validate Data
    loader = BinanceDataLoader(config.data.raw_path)
    df = loader.load_raw_klines()

    print("\n--- Data Ingestion Successful ---")
    print(f"Total Rows Loaded: {len(df):,}")
    print(f"Time Range: {df['datetime'].min()} to {df['datetime'].max()}")
    print("\nFirst 3 Rows Preview:")
    print(df[["datetime", "open", "high", "low", "close", "volume"]].head(3))


if __name__ == "__main__":
    main()