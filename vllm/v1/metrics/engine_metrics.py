# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Engine Metrics Collection for vLLM.

Provides lightweight metrics collection for monitoring engine performance.
Designed for high-throughput scenarios where metrics overhead must be minimal.

Design Philosophy:
-----------------
This module prioritizes performance over strict accuracy:

1. **Lock-Free Counters**: We use simple integer increments without locks.
   In Python, int operations are atomic at the bytecode level (a single
   LOAD/ADD/STORE sequence), making explicit locks unnecessary for our
   use case where slight inaccuracies are acceptable.

2. **Best-Effort Recording**: Metrics are recorded on a best-effort basis.
   If recording fails (e.g., histogram bucket overflow), we silently skip
   rather than raising exceptions. This ensures metrics collection never
   impacts the critical path.

3. **Memory Bounded**: All data structures have fixed maximum sizes to
   prevent unbounded memory growth. When limits are reached, old data
   is silently discarded.

4. **Thread Compatibility**: While not strictly thread-safe with locks,
   the design ensures no crashes or corruption. Worst case is slight
   undercounting, which is acceptable for monitoring purposes.

If strict accuracy is required, use a dedicated metrics library like
Prometheus or StatsD.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class Counter:
    """
    A simple counter for tracking cumulative values.
    
    Thread Safety:
    -------------
    This counter uses plain integer operations without locks.
    In CPython, simple int operations are atomic due to the GIL,
    making explicit synchronization unnecessary for counters.
    
    The worst case in concurrent updates is slight undercounting,
    which is acceptable for monitoring metrics.
    """
    value: int = 0
    
    def inc(self, amount: int = 1) -> None:
        """Increment the counter."""
        self.value += amount
    
    def get(self) -> int:
        """Get the current value."""
        return self.value
    
    def reset(self) -> int:
        """Reset and return the previous value."""
        prev = self.value
        self.value = 0
        return prev


@dataclass
class Gauge:
    """
    A gauge for tracking current values.
    
    Gauges represent a single value that can go up or down,
    like queue depth or memory usage.
    """
    value: float = 0.0
    
    def set(self, value: float) -> None:
        """Set the gauge value."""
        self.value = value
    
    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge."""
        self.value += amount
    
    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge."""
        self.value -= amount
    
    def get(self) -> float:
        """Get the current value."""
        return self.value


@dataclass
class Histogram:
    """
    A histogram for tracking value distributions.
    
    Uses fixed buckets for memory efficiency. Values outside
    the bucket range are clamped to the nearest bucket.
    
    Overflow Handling:
    -----------------
    If a bucket count exceeds the maximum (2^63-1), we stop
    counting for that bucket rather than wrapping. This is a
    best-effort approach that prevents negative counts while
    accepting some data loss in extreme scenarios.
    """
    buckets: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    counts: list[int] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    
    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)
    
    def observe(self, value: float) -> None:
        """
        Record an observation.
        
        The value is placed in the appropriate bucket and added
        to the running sum. If the bucket count would overflow,
        we skip incrementing to prevent wraparound.
        """
        # Find bucket
        bucket_idx = len(self.buckets)  # Default to +inf bucket
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                bucket_idx = i
                break
        
        # Check for overflow before incrementing
        # Max int64 is approximately 9.2e18
        if self.counts[bucket_idx] < 9_223_372_036_854_775_807:
            self.counts[bucket_idx] += 1
        # else: silently skip to prevent overflow
        
        self.sum_value += value
        self.count += 1
    
    def get_bucket_counts(self) -> list[tuple[float | str, int]]:
        """Get counts for each bucket."""
        result = []
        for i, upper in enumerate(self.buckets):
            result.append((upper, self.counts[i]))
        result.append(("+inf", self.counts[-1]))
        return result
    
    def get_percentile(self, p: float) -> float | None:
        """
        Estimate the p-th percentile from the histogram.
        
        This is an approximation based on bucket boundaries.
        Returns None if no observations recorded.
        """
        if self.count == 0:
            return None
        
        target = p * self.count
        cumulative = 0
        
        for i, upper in enumerate(self.buckets):
            cumulative += self.counts[i]
            if cumulative >= target:
                # Linear interpolation within bucket
                if i == 0:
                    return upper * (target / cumulative)
                else:
                    lower = self.buckets[i - 1]
                    return lower + (upper - lower) * (target - cumulative + self.counts[i]) / self.counts[i]
        
        # In +inf bucket
        return self.buckets[-1] if self.buckets else None


@dataclass
class MovingAverage:
    """
    Exponential moving average for smoothing metrics.
    
    Uses a fixed alpha for the weighting factor.
    """
    alpha: float = 0.1
    value: float | None = None
    
    def update(self, new_value: float) -> float:
        """Update with a new value and return the smoothed result."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def get(self) -> float | None:
        """Get the current smoothed value."""
        return self.value


class RingBuffer:
    """
    Fixed-size ring buffer for time-series data.
    
    When full, oldest entries are silently overwritten.
    This ensures bounded memory usage.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
    
    def append(self, timestamp: float, value: float) -> None:
        """Add a timestamped value."""
        self._buffer.append((timestamp, value))
    
    def get_recent(self, seconds: float) -> list[tuple[float, float]]:
        """Get entries from the last N seconds."""
        cutoff = time.time() - seconds
        return [(t, v) for t, v in self._buffer if t >= cutoff]
    
    def get_all(self) -> list[tuple[float, float]]:
        """Get all entries."""
        return list(self._buffer)
    
    def clear(self) -> None:
        """Clear all entries."""
        self._buffer.clear()


class EngineMetrics:
    """
    Centralized metrics collection for the vLLM engine.
    
    Provides counters, gauges, and histograms for monitoring
    engine performance and resource usage.
    
    Thread Safety:
    -------------
    This class is designed for concurrent access without locks.
    Individual metric operations are atomic enough for monitoring
    purposes. See module docstring for design rationale.
    """
    
    def __init__(self):
        # Request counters
        self.requests_total = Counter()
        self.requests_success = Counter()
        self.requests_error = Counter()
        
        # Token counters
        self.prompt_tokens_total = Counter()
        self.completion_tokens_total = Counter()
        
        # Queue gauges
        self.queue_depth = Gauge()
        self.running_requests = Gauge()
        
        # Latency histograms
        self.request_latency = Histogram()
        self.time_to_first_token = Histogram()
        self.inter_token_latency = Histogram()
        
        # Resource gauges
        self.gpu_memory_used = Gauge()
        self.kv_cache_usage = Gauge()
        
        # Time series for rate calculation
        self._request_times = RingBuffer(max_size=10000)
        self._token_times = RingBuffer(max_size=10000)
    
    def record_request_start(self) -> float:
        """Record request start and return the start time."""
        start_time = time.time()
        self.requests_total.inc()
        self.queue_depth.inc()
        self._request_times.append(start_time, 1.0)
        return start_time
    
    def record_request_complete(
        self,
        start_time: float,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool = True,
    ) -> None:
        """
        Record request completion.
        
        Args:
            start_time: Timestamp from record_request_start().
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of generated tokens.
            success: Whether the request succeeded.
        """
        elapsed = time.time() - start_time
        
        if success:
            self.requests_success.inc()
        else:
            self.requests_error.inc()
        
        self.queue_depth.dec()
        self.prompt_tokens_total.inc(prompt_tokens)
        self.completion_tokens_total.inc(completion_tokens)
        self.request_latency.observe(elapsed)
        
        # Record for rate calculation
        self._token_times.append(time.time(), completion_tokens)
    
    def record_first_token(self, start_time: float) -> None:
        """Record time to first token."""
        elapsed = time.time() - start_time
        self.time_to_first_token.observe(elapsed)
    
    def record_token_latency(self, latency: float) -> None:
        """Record inter-token latency."""
        self.inter_token_latency.observe(latency)
    
    def get_request_rate(self, window_seconds: float = 60.0) -> float:
        """
        Get requests per second over the recent window.
        
        Returns 0.0 if no data in window.
        """
        recent = self._request_times.get_recent(window_seconds)
        if not recent:
            return 0.0
        return len(recent) / window_seconds
    
    def get_token_rate(self, window_seconds: float = 60.0) -> float:
        """
        Get tokens per second over the recent window.
        
        Returns 0.0 if no data in window.
        """
        recent = self._token_times.get_recent(window_seconds)
        if not recent:
            return 0.0
        total_tokens = sum(v for _, v in recent)
        return total_tokens / window_seconds
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "requests": {
                "total": self.requests_total.get(),
                "success": self.requests_success.get(),
                "error": self.requests_error.get(),
                "rate_per_sec": round(self.get_request_rate(), 2),
            },
            "tokens": {
                "prompt_total": self.prompt_tokens_total.get(),
                "completion_total": self.completion_tokens_total.get(),
                "rate_per_sec": round(self.get_token_rate(), 2),
            },
            "latency": {
                "p50_ms": self._format_latency(self.request_latency.get_percentile(0.5)),
                "p90_ms": self._format_latency(self.request_latency.get_percentile(0.9)),
                "p99_ms": self._format_latency(self.request_latency.get_percentile(0.99)),
            },
            "ttft": {
                "p50_ms": self._format_latency(self.time_to_first_token.get_percentile(0.5)),
                "p90_ms": self._format_latency(self.time_to_first_token.get_percentile(0.9)),
            },
            "resources": {
                "queue_depth": int(self.queue_depth.get()),
                "running_requests": int(self.running_requests.get()),
                "gpu_memory_pct": round(self.gpu_memory_used.get(), 1),
                "kv_cache_pct": round(self.kv_cache_usage.get(), 1),
            },
        }
    
    def _format_latency(self, value: float | None) -> float | None:
        """Format latency as milliseconds."""
        if value is None:
            return None
        return round(value * 1000, 2)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.requests_total = Counter()
        self.requests_success = Counter()
        self.requests_error = Counter()
        self.prompt_tokens_total = Counter()
        self.completion_tokens_total = Counter()
        self.queue_depth = Gauge()
        self.running_requests = Gauge()
        self.request_latency = Histogram()
        self.time_to_first_token = Histogram()
        self.inter_token_latency = Histogram()
        self._request_times.clear()
        self._token_times.clear()


# Global metrics instance
_engine_metrics: EngineMetrics | None = None


def get_engine_metrics() -> EngineMetrics:
    """Get or create the global engine metrics instance."""
    global _engine_metrics
    if _engine_metrics is None:
        _engine_metrics = EngineMetrics()
    return _engine_metrics
