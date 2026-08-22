import sys
import argparse
import logging
from pathlib import Path

from ml_crypto.config import get_config
from ml_crypto.pipeline.runner import PipelineRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('main')

def main() -> None:
    parser = argparse.ArgumentParser(
        description="unified pipeline for ml-crypto",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/config.yaml",
        help="path to YAML configuration file"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=['train', 'drift-retrain'],
        default='train',
        help="pipeline execution mode: 'train' (always run) or 'drift-retrain' (run only if feature drift detected)"
    )
    parser.add_argument("--data-path",
        type=str,
        default=None,
        help="optional raw data path override"
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"configuration path not found: {config_path}")
        sys.exit(1)
    
    try:
        config = get_config(str(config_path))
        runner = PipelineRunner(config=config)
        
        result = runner.run(mode=args.mode, data_path_override=args.data_path)
        
        print("\n" + "=" * 65)
        print(f" PIPELINE EXECUTION SUMMARY | MODE: {args.mode.upper()}")
        print("=" * 65)
        print(f" Project               : {config.system.project_name}")
        print(f" Execution Status      : {result['status'].upper()}")

        if result["status"] == "success":
            manifest = result["manifest"]
            metrics = manifest["metrics"]
            print(f" Run ID                : {manifest['run_id']}")
            print(f" Best Iteration        : {metrics.get('best_iteration', 'N/A')}")
            print(f" Train RMSE            : {metrics['train_rmse']:.6f}")
            print(f" Validation RMSE       : {metrics['val_rmse']:.6f}")
            print(f" Val Directional Acc   : {metrics['directional_accuracy'] * 100:.2f}%")
            print(f" Test Directional Acc  : {metrics['test_directional_accuracy'] * 100:.2f}% (Holdout)")
            print(f" Information Ratio     : {metrics.get('information_ratio', 0.0):.3f}")
            print(f" Promoted to Prod      : {manifest['is_promoted_to_production']}")
        else:
            print(f" Reason                : {result['reason']}")

        print(f" Drift Checked         : {args.mode == 'drift-retrain'}")
        print("=" * 65 + "\n")

        sys.exit(0)
    
    except Exception as e:
        logger.critical(f"pipeline crashed with unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()