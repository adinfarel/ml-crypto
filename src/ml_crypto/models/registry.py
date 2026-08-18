import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class ModelRegistry:
    '''Manage model.'''
    
    def __init__(self, models_dir: Path, runs_dir: Path):
        self.models_dir = models_dir
        self.runs_dir = runs_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def save_run(self, run_id: str, model: Any, manifest: Dict[str, Any]) -> Path:
        '''Save trained model binary.'''
        model_path = self.models_dir / f"model_{run_id}.pkl"
        joblib.dump(model, model_path)
        
        manifest_path = self.runs_dir / f"run_{run_id}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return model_path
    
    def promote_if_better(self, run_id: str, current_metric: float, metric_name: str = "directional_accuracy") -> bool:
        '''Evaluation; if new model better then become production model.'''
        prod_manifest_path = self.models_dir / "production_manifest.json"
        
        if not prod_manifest_path.exists():
            self._set_production(run_id)
            return True

        with open(prod_manifest_path, 'r') as f:
            prod_manifest = json.load(f)
        
        prev_metric = prod_manifest.get("metrics", {}).get(metric_name, -float("-inf"))
        
        # must beat prev metrics
        if current_metric > prev_metric:
            self._set_production(run_id)
            return True
        
        return False
    
    def _set_production(self, run_id: str) -> None:
        source_model = self.models_dir / f"model_{run_id}.pkl"
        prod_model = self.models_dir / "production_model.pkl"
        
        # replace
        joblib.dump(joblib.load(source_model), prod_model)
        
        source_manifest = self.runs_dir / f"run_{run_id}.json"
        with open(source_manifest, "r") as f:
            manifest = json.load(f)

        with open(self.models_dir / "production_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)