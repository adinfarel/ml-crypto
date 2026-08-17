from src.config import get_config
from src.utils import set_seed, calculate_file_hash


def main():
    # Load typed configuration
    config = get_config("config/config.yaml")
    
    # Set seed for reproducibility
    set_seed(config.system.random_seed)
    
    print(f"=== Project: {config.system.project_name} ===")
    print(f"Random Seed set to: {config.system.random_seed}")
    print(f"Target Column: {config.data.target_column}")
    print(f"Raw Data Path: {config.data.raw_path}")
    
    # Check data hash if raw file exists
    data_hash = calculate_file_hash(config.data.raw_path)
    print(f"Raw Data MD5 Hash: {data_hash}")


if __name__ == "__main__":
    main()