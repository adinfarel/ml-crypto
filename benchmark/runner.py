import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmark.metrics import (
    BenchmarkStats,
    benchmark_stage,
)

class BenchmarkRunner:
    
    def __init__(
        self,
        name: str,
        output_dir: Path = Path('benchmark/results')
    ):
        self.name = name
        self.output_dir = output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stages: list[tuple[str, Callable[[], Any]]] = []
    
    def add_stage(
        self,
        name: str,
        func: Callable[[], Any]
    ) -> None:
        self.stages.append((name, func))
    
    def run(
        self,
        *,
        warmup: int = 1,
        iterations: int = 5
    ) -> dict[str, Any]:
        
        results: dict[str, BenchmarkStats] = {}
        
        print("="*70)
        print(f"BENCHMARK: {self.name}")
        print("="*70)
        
        for name, func in self.stages:
            
            print(f"\n[{name!r}]")
            
            _, stats = benchmark_stage(
                name,
                func,
                warmup=warmup,
                iterations=iterations,
            )
            
            results[name] = stats
            
            self._print_stats(stats)
        
        report = self._build_report(results)
        
        output_path = (
            self.output_dir / f"{self.name}.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\n" + "=" * 70)
        print("Report saved to :", str(output_path))
        print("=" * 70)

    @staticmethod
    def _print_stats(
        stats: BenchmarkStats,
    ) -> None:
        print(
            f"median : {stats.median_s:.4f}s"
        )

        print(
            f"mean   : {stats.mean_s:.4f}s"
        )

        print(
            f"p95    : {stats.p95_s:.4f}s"
        )

        print(
            f"min    : {stats.min_s:.4f}s"
        )
        
        print(
            f"max    : {stats.max_s:.4f}s"
        )

        print(
            f"memory : {stats.memory_delta_mb:+.2f} MB"
        )

        print(
            f"peak   : {stats.peak_memory_delta_mb:+.2f} MB"
        )
        
    def _build_report(
        self,
        results: dict[str, BenchmarkStats]
    ) -> dict[str, Any]:
        return {
            "benchmark": self.name,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "git_commit": self._get_git_commit(),
            "results": {
                name: stats.to_dict()
                for name, stats in results.items()
            }
        }
    
    @staticmethod
    def _get_git_commit() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except (
            subprocess.CalledProcessError,
            FileNotFoundError
        ):
            return None