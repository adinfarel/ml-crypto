import gc
import os
import time
import threading
import psutil
from dataclasses import dataclass, asdict
from statistics import median
from typing import Any, Callable, Optional

_PROCESS = psutil.Process(os.getpid())

@dataclass
class StageResult:
    name: str
    
    # latency
    time_s: float
    
    # memory
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_mb: float
    peak_memory_delta_mb: float
    
    def to_dict(self,) -> dict[str, Any]:
        return asdict(self)

@dataclass
class BenchmarkStats:
    name: str
    iterations: int
    
    median_s: float
    mean_s: float
    min_s: float
    max_s: float
    p95_s: float
    
    memory_delta_mb: float
    peak_memory_delta_mb: float
    
    def to_dict(self,) -> dict[str, Any]:
        return asdict(self)

def get_memory_mb() -> float:
    '''Current memory process (RSS) of this python process.'''
    return _PROCESS.memory_info().rss / (1024**2) # mb

class _PeakMemorySampler:
    
    def __init__(self, interval_s: float = 0.005):
        self.interval_s = interval_s
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self.peak_mb = 0.0
    
    def _sample(self,) -> None:
        while not self._stop_event.is_set():
            current = get_memory_mb()
            
            if current > self.peak_mb:
                self.peak_mb = current
            
            self._stop_event.wait(self.interval_s)
    
    def start(self) -> None:
        self.peak_mb = get_memory_mb()
        
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        
        self._thread.start()
    
    def stop(self) -> None:
        self._stop_event.set()
        
        if self._thread is not None:
            self._thread.join()
        
        self.peak_mb = max(
            self.peak_mb,
            get_memory_mb()
        )
        
        return self.peak_mb

def measure_stage(
    name: str,
    func: Callable[[], Any],
    *,
    gc_before: bool = True,
    memory_sampling_interval_s: float = 0.005,
) -> tuple[Any, StageResult]:
    if gc_before:
        gc.collect()
    
    memory_before = get_memory_mb()
    
    sampler = _PeakMemorySampler(
        interval_s=memory_sampling_interval_s,
    )
    
    sampler.start()
    start = time.perf_counter()
    
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start
        peak_memory = sampler.stop()
    
    memory_after = get_memory_mb()
    
    stage_result = StageResult(
        name=name,
        time_s=elapsed,

        memory_before_mb=memory_before,
        memory_after_mb=memory_after,

        memory_delta_mb=memory_after - memory_before,

        peak_memory_mb=peak_memory,
        peak_memory_delta_mb=peak_memory - memory_before,
    )
    
    return result, stage_result

def benchmark_stage(
    name: str,
    func: Callable[[], Any],
    *,
    warmup: int = 1,
    iterations: int = 5,
) -> tuple[list[Any], BenchmarkStats]:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    
    # warmup
    for _ in range(warmup):
        func()
    
    times: list[float] = []
    memory_deltas: list[float] = []
    peak_memory_deltas: list[float] = []
    
    results: list[Any] = []
    
    for _ in range(iterations):
        result, measurement = measure_stage(
            name,
            func,
        )
    
        results.append(result)
        
        times.append(measurement.time_s)
        memory_deltas.append(measurement.memory_delta_mb)
        peak_memory_deltas.append(measurement.peak_memory_delta_mb)
    
    sorted_times = sorted(times)
    
    p95_index = min(
        len(sorted_times) - 1,
        max(0, int(0.95 * len(sorted_times)) - 1)
    )
    
    stats = BenchmarkStats(
        name=name,
        iterations=iterations,

        median_s=median(times),
        mean_s=sum(times) / len(times),
        min_s=min(times),
        max_s=max(times),
        p95_s=sorted_times[p95_index],

        memory_delta_mb=median(memory_deltas),
        peak_memory_delta_mb=median(peak_memory_deltas),
    )

    return results, stats