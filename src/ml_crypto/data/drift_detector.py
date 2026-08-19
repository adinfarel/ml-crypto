import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List

import scipy

class DriftDetector:
    '''Calculate PSI (population stability index) and KS (kolmogorov-smirnov) test for feature drift.'''
    
    __slots__ = ("num_buckets",)
    
    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets
        
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:       
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        percentiles = np.linspace(0, 100, self.num_buckets + 1)
        bins = np.percentile(reference, percentiles)
        bins = np.unique(bins)
        
        if len(bins) <= 1:
            return 0.0
        
        bins[0] = min(bins[0], current.min()) - 1e-5
        bins[-1] = max(bins[-1], current.max()) + 1e-5
        
        ref_counts, _ = np.histogram(reference, bins=bins)
        curr_counts, _ = np.histogram(current, bins=bins)
        
        eps = 1e-4
        ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
        curr_pct = (curr_counts + eps) / (len(current) + eps * len(curr_counts))
        # PSI formula:
        # sum(actual% - expected%) * ln(actual% / expected%)
        psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi_value)
    
    def detect_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, Any]:
        drift_results = {}
        overall_drift_flag = False
        
        for col in feature_cols:
            ref_data = reference_df[col].dropna().values
            curr_data = current_df[col].dropna().values
            
            psi_val = self.calculate_psi(ref_data, curr_data)
            
            ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
            # drift threshold: PSI > 0.25 or KS p-value < 0.01
            is_drifted = bool(psi_val > 0.25 or p_value < 0.01)
            if is_drifted:
                overall_drift_flag = True
                
            drift_results[col] = {
                "psi": round(psi_val, 4),
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 6),
                "drift_detected": is_drifted,
            }

        return {
            "has_drift": overall_drift_flag,
            "feature_report": drift_results,
        }