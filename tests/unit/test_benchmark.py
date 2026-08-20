import time

from benchmark.runner import BenchmarkRunner

def slow_func():
    time.sleep(0.1)
    return 42

def test_benchmark():
    runner = BenchmarkRunner(name="test_benchmark")
    
    runner.add_stage(
        "slow_func",
        slow_func,
    )
    
    report = runner.run(
        warmup=2,
        iterations=5
    )