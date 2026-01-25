# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA Adapter Loading Utilities.

Provides utilities for loading, caching, and managing LoRA adapters
in a multi-tenant environment.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


@dataclass
class LoadedAdapter:
    """Metadata for a loaded LoRA adapter."""
    name: str
    path: str
    int_id: int
    load_time: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)


class LoRAAdapterLoader:
    """
    Manages loading and caching of LoRA adapters.
    
    This class provides a layer on top of the engine's LoRA loading
    to add caching, validation, and multi-tenant isolation features.
    
    Thread Safety:
    -------------
    The loader uses per-adapter locks to prevent race conditions when
    multiple requests try to load the same adapter concurrently. This
    ensures that:
    - Only one request performs the actual load
    - Other concurrent requests wait and receive the cached result
    - Unique adapter IDs are correctly assigned
    """
    
    def __init__(
        self,
        engine_client: Any,
        max_loaded_adapters: int = 100,
        cache_ttl_seconds: float = 3600.0,
    ):
        """
        Initialize the LoRA adapter loader.
        
        Args:
            engine_client: The vLLM engine client.
            max_loaded_adapters: Maximum number of adapters to keep loaded.
            cache_ttl_seconds: TTL for adapter cache entries.
        """
        self.engine_client = engine_client
        self.max_loaded_adapters = max_loaded_adapters
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Cache of loaded adapters: name -> LoadedAdapter
        self._loaded_adapters: dict[str, LoadedAdapter] = {}
        
        # Counter for generating unique adapter IDs
        self._id_counter = 0
        
        # Per-adapter locks to prevent race conditions
        self._adapter_locks: dict[str, asyncio.Lock] = {}
        self._lock_creation_lock = asyncio.Lock()
    
    def _next_adapter_id(self) -> int:
        """Get the next unique adapter ID."""
        self._id_counter += 1
        return self._id_counter
    
    async def _get_adapter_lock(self, adapter_name: str) -> asyncio.Lock:
        """Get or create a lock for the given adapter."""
        async with self._lock_creation_lock:
            if adapter_name not in self._adapter_locks:
                self._adapter_locks[adapter_name] = asyncio.Lock()
            return self._adapter_locks[adapter_name]
    
    async def load_adapter(
        self,
        adapter_name: str,
        adapter_path: str,
        base_model_name: str | None = None,
        force_reload: bool = False,
    ) -> LoRARequest:
        """
        Load a LoRA adapter, using cache if available.
        
        This method uses per-adapter locking to prevent race conditions
        when multiple requests try to load the same adapter concurrently.
        Only one request will perform the actual load; others will wait
        and receive the cached result.
        
        Args:
            adapter_name: Unique name for the adapter.
            adapter_path: Path to the adapter files.
            base_model_name: Optional base model name.
            force_reload: If True, reload even if cached.
        
        Returns:
            A LoRARequest for the loaded adapter.
        """
        # Get the per-adapter lock to prevent race conditions
        adapter_lock = await self._get_adapter_lock(adapter_name)
        
        async with adapter_lock:
            # Check if already loaded (inside lock to prevent TOCTOU)
            if not force_reload and adapter_name in self._loaded_adapters:
                cached = self._loaded_adapters[adapter_name]
                cached.access_count += 1
                cached.last_access = time.time()
                logger.debug(
                    "Using cached LoRA adapter '%s' (hits=%d)",
                    adapter_name, cached.access_count
                )
                return LoRARequest(
                    lora_name=adapter_name,
                    lora_int_id=cached.int_id,
                    lora_path=cached.path,
                    base_model_name=base_model_name,
                )
            
            # Validate path exists
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(
                    f"LoRA adapter path does not exist: {adapter_path}"
                )
            
            # Generate a new ID if not cached, or reuse existing ID
            if adapter_name in self._loaded_adapters:
                adapter_id = self._loaded_adapters[adapter_name].int_id
            else:
                adapter_id = self._next_adapter_id()
            
            # Create the request
            lora_request = LoRARequest(
                lora_name=adapter_name,
                lora_int_id=adapter_id,
                lora_path=adapter_path,
                base_model_name=base_model_name,
                load_inplace=force_reload,
            )
            
            # Load into engine (inside lock to ensure atomicity)
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                logger.error(
                    "Failed to load LoRA adapter '%s': %s",
                    adapter_name, e
                )
                raise
            
            # Update cache
            # Evict old entries if at capacity
            await self._evict_if_needed()
            
            self._loaded_adapters[adapter_name] = LoadedAdapter(
                name=adapter_name,
                path=adapter_path,
                int_id=adapter_id,
                load_time=time.time(),
            )
            
            logger.info(
                "Loaded LoRA adapter '%s' with ID %d",
                adapter_name, adapter_id
            )
            
            return lora_request
    
    async def get_adapter(
        self,
        adapter_name: str,
    ) -> LoRARequest | None:
        """
        Get a loaded adapter by name.
        
        Args:
            adapter_name: Name of the adapter.
        
        Returns:
            LoRARequest if loaded, None otherwise.
        """
        if adapter_name not in self._loaded_adapters:
            return None
        
        cached = self._loaded_adapters[adapter_name]
        cached.access_count += 1
        cached.last_access = time.time()
        
        return LoRARequest(
            lora_name=adapter_name,
            lora_int_id=cached.int_id,
            lora_path=cached.path,
        )
    
    async def unload_adapter(
        self,
        adapter_name: str,
    ) -> bool:
        """
        Unload a LoRA adapter.
        
        Args:
            adapter_name: Name of the adapter to unload.
        
        Returns:
            True if unloaded, False if not found.
        """
        if adapter_name not in self._loaded_adapters:
            logger.warning(
                "Attempted to unload unknown adapter: %s",
                adapter_name
            )
            return False
        
        del self._loaded_adapters[adapter_name]
        logger.info("Unloaded LoRA adapter '%s'", adapter_name)
        return True
    
    async def _evict_if_needed(self) -> None:
        """Evict least recently used adapters if at capacity."""
        while len(self._loaded_adapters) >= self.max_loaded_adapters:
            # Find LRU adapter
            lru_name = min(
                self._loaded_adapters.keys(),
                key=lambda n: self._loaded_adapters[n].last_access
            )
            await self.unload_adapter(lru_name)
    
    def get_stats(self) -> dict[str, Any]:
        """Get loader statistics."""
        return {
            "loaded_count": len(self._loaded_adapters),
            "max_loaded": self.max_loaded_adapters,
            "adapters": {
                name: {
                    "path": adapter.path,
                    "id": adapter.int_id,
                    "access_count": adapter.access_count,
                    "load_time": adapter.load_time,
                    "last_access": adapter.last_access,
                }
                for name, adapter in self._loaded_adapters.items()
            }
        }


async def resolve_adapter_path(
    adapter_name: str,
    search_paths: list[str] | None = None,
) -> str | None:
    """
    Resolve an adapter name to a filesystem path.
    
    Searches common locations for the adapter directory.
    
    Args:
        adapter_name: Name or partial path of the adapter.
        search_paths: Optional list of paths to search.
    
    Returns:
        Full path to adapter if found, None otherwise.
    """
    # If it's already a valid path, return it
    if os.path.exists(adapter_name):
        return adapter_name
    
    # Default search paths
    if search_paths is None:
        search_paths = [
            os.path.expanduser("~/.cache/vllm/lora"),
            "/opt/vllm/lora",
            "./lora_adapters",
        ]
    
    for base_path in search_paths:
        candidate = os.path.join(base_path, adapter_name)
        if os.path.exists(candidate):
            return candidate
    
    return None


def validate_adapter_config(adapter_path: str) -> tuple[bool, str | None]:
    """
    Validate that an adapter directory contains required files.
    
    Args:
        adapter_path: Path to the adapter directory.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    path = Path(adapter_path)
    
    if not path.exists():
        return False, f"Adapter path does not exist: {adapter_path}"
    
    if not path.is_dir():
        return False, f"Adapter path is not a directory: {adapter_path}"
    
    # Check for required files
    required_files = ["adapter_config.json"]
    for filename in required_files:
        if not (path / filename).exists():
            return False, f"Missing required file: {filename}"
    
    return True, None
